import * as googleTTS from "google-tts-api";
import * as os from "os";
import * as path from "path";
import * as https from "https";
import * as child_process from "child_process";
import { existsSync, writeFileSync, unlinkSync } from "fs";
import { getPreferenceValues } from "@raycast/api";
import { request, ProxyAgent } from "undici";
import { LanguageCode, supportedLanguagesByCode } from "./languages";
import { LanguageCodeSet } from "./types";
import { HttpsProxyAgent } from "https-proxy-agent";

export const AUTO_DETECT = "auto";

export type SimpleTranslateResult = {
  originalText: string;
  translatedText: string;
  pronunciationText?: string;
  langFrom: LanguageCode;
  langTo: LanguageCode;
  proxy?: string;
};

export class TranslateError extends Error {}

type OpenAICompatiblePreferences = {
  endpoint?: string;
  model?: string;
  accessToken?: string;
  proxy?: string;
};

const DEFAULT_ENDPOINT = "https://api.openai.com/v1";
const DEFAULT_MODEL = "gpt-4o-mini";

const normalizeCode = (value: string) => value.trim().replace(/_/g, "-");

const languageNameToCode = new Map(
  Object.values(supportedLanguagesByCode)
    .filter((language) => language.code !== AUTO_DETECT)
    .map((language) => [language.name.toLowerCase(), language.code] as const),
);

const toChatCompletionsURL = (endpoint: string) => {
  const normalized = endpoint.trim().replace(/\/+$/, "");
  if (!normalized) {
    return `${DEFAULT_ENDPOINT}/chat/completions`;
  }
  if (normalized.endsWith("/chat/completions")) {
    return normalized;
  }
  if (normalized.endsWith("/v1")) {
    return `${normalized}/chat/completions`;
  }
  return `${normalized}/v1/chat/completions`;
};

const extractJsonObject = (text: string): Record<string, unknown> | null => {
  const firstBraceIndex = text.indexOf("{");
  const lastBraceIndex = text.lastIndexOf("}");
  if (firstBraceIndex < 0 || lastBraceIndex < 0 || lastBraceIndex <= firstBraceIndex) {
    return null;
  }

  const jsonSlice = text.slice(firstBraceIndex, lastBraceIndex + 1);
  try {
    return JSON.parse(jsonSlice) as Record<string, unknown>;
  } catch {
    return null;
  }
};

const extractContentText = (content: unknown): string => {
  if (typeof content === "string") {
    return content.trim();
  }

  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") {
          return part;
        }
        if (part && typeof part === "object" && "text" in part && typeof part.text === "string") {
          return part.text;
        }
        return "";
      })
      .join("")
      .trim();
  }

  return "";
};

const resolveDetectedLanguageCode = (detectedCode: unknown, fallback: LanguageCode): LanguageCode => {
  if (fallback !== AUTO_DETECT) {
    return fallback;
  }

  if (typeof detectedCode !== "string") {
    return AUTO_DETECT;
  }

  const normalized = normalizeCode(detectedCode);
  if (normalized in supportedLanguagesByCode) {
    return normalized as LanguageCode;
  }

  const lowerCased = normalized.toLowerCase();
  const matchedCode = Object.keys(supportedLanguagesByCode).find((code) => code.toLowerCase() === lowerCased);
  if (matchedCode) {
    return matchedCode as LanguageCode;
  }

  const byName = languageNameToCode.get(lowerCased);
  if (byName) {
    return byName;
  }

  return AUTO_DETECT;
};

const requestLLMTranslation = async (
  text: string,
  options: LanguageCodeSet,
): Promise<Pick<SimpleTranslateResult, "translatedText" | "pronunciationText" | "langFrom">> => {
  const preferences = getPreferenceValues<OpenAICompatiblePreferences>();

  const endpoint = preferences.endpoint?.trim() || DEFAULT_ENDPOINT;
  const model = preferences.model?.trim() || DEFAULT_MODEL;
  const accessToken = preferences.accessToken?.trim();
  const proxy = options.proxy ?? preferences.proxy?.trim();

  if (!accessToken) {
    const error = new TranslateError("Missing access token");
    error.name = "Configuration Error";
    error.message = "Please set Access Token in extension preferences.";
    throw error;
  }

  const sourceLanguage = supportedLanguagesByCode[options.langFrom];
  const targetLanguage = supportedLanguagesByCode[options.langTo[0]];

  if (!targetLanguage) {
    const error = new TranslateError("Unsupported target language");
    error.name = "Configuration Error";
    error.message = `Unsupported target language: ${options.langTo[0]}`;
    throw error;
  }

  const sourceLabel =
    options.langFrom === AUTO_DETECT
      ? "auto-detect from input text"
      : `${sourceLanguage?.name ?? options.langFrom} (${options.langFrom})`;
  const targetLabel = `${targetLanguage.name} (${targetLanguage.code})`;

  const responseFormat = `Return only valid JSON with keys:
{"translation":"...", "detected_source_language_code":"...", "pronunciation":"..."}
Rules:
- "translation" is required.
- "detected_source_language_code" must be a language code only when source is auto-detect; otherwise repeat "${options.langFrom}".
- "pronunciation" should be transliteration text when useful, otherwise empty string.
- No markdown, no code fences, no extra keys.`;

  const systemPrompt =
    "You are a precise translation engine. Keep meaning, tone, and formatting. Do not explain the translation.";
  const userPrompt = `Source language: ${sourceLabel}
Target language: ${targetLabel}
Text to translate:
${text}

${responseFormat}`;

  const requestOptions: Parameters<typeof request>[1] = {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${accessToken}`,
    },
    body: JSON.stringify({
      model,
      temperature: 0,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
    }),
  };

  if (proxy) {
    requestOptions.dispatcher = new ProxyAgent(proxy);
  }

  const response = await request(toChatCompletionsURL(endpoint), requestOptions);
  const responseText = await response.body.text();

  let payload: Record<string, unknown> = {};
  try {
    payload = JSON.parse(responseText) as Record<string, unknown>;
  } catch {
    // Keep payload empty and fall through to generic error below.
  }

  if (response.statusCode >= 400) {
    const payloadError = payload.error;
    const apiError =
      payloadError &&
      typeof payloadError === "object" &&
      typeof (payloadError as { message?: unknown }).message === "string"
        ? ((payloadError as { message: string }).message ?? "")
        : responseText;
    const error = new TranslateError("Translation request failed");
    error.name = `HTTP ${response.statusCode}`;
    error.message = apiError || "Unknown API error";
    throw error;
  }

  const choices = Array.isArray(payload.choices) ? payload.choices : [];
  const firstChoice = choices[0] as Record<string, unknown> | undefined;
  const message = firstChoice?.message as Record<string, unknown> | undefined;
  const rawContent = extractContentText(message?.content);

  if (!rawContent) {
    const error = new TranslateError("Empty model response");
    error.name = "Invalid Response";
    error.message = "Model returned empty content.";
    throw error;
  }

  const parsedContent = extractJsonObject(rawContent);
  const translatedText =
    parsedContent && typeof parsedContent.translation === "string"
      ? parsedContent.translation.trim()
      : rawContent.trim();

  if (!translatedText) {
    const error = new TranslateError("Missing translation content");
    error.name = "Invalid Response";
    error.message = "Model response does not include translation text.";
    throw error;
  }

  const pronunciationText =
    parsedContent && typeof parsedContent.pronunciation === "string" ? parsedContent.pronunciation.trim() : "";
  const langFrom = resolveDetectedLanguageCode(parsedContent?.detected_source_language_code, options.langFrom);

  return {
    translatedText,
    pronunciationText,
    langFrom,
  };
};

export async function simpleTranslate(text: string, options: LanguageCodeSet): Promise<SimpleTranslateResult> {
  try {
    if (!text) {
      return {
        originalText: text,
        translatedText: "",
        pronunciationText: "",
        langFrom: options.langFrom,
        langTo: options.langTo[0],
      };
    }

    const translated = await requestLLMTranslation(text, options);

    return {
      originalText: text,
      translatedText: translated.translatedText,
      pronunciationText: translated.pronunciationText,
      langFrom: translated.langFrom,
      langTo: options.langTo[0],
    };
  } catch (err) {
    if (err instanceof Error) {
      const error = new TranslateError();
      error.name = err.name;
      error.message = err.message;
      throw error;
    }

    throw err;
  }
}

export async function doubleWayTranslate(text: string, options: LanguageCodeSet) {
  if (!text) {
    return [];
  }

  if (options.langFrom === AUTO_DETECT) {
    const translated1 = await simpleTranslate(text, {
      langFrom: options.langFrom,
      langTo: options.langTo,
      proxy: options.proxy,
    });

    if (translated1?.langFrom && translated1.langFrom !== AUTO_DETECT) {
      const translated2 = await simpleTranslate(translated1.translatedText, {
        langFrom: options.langTo[0],
        langTo: [translated1.langFrom],
        proxy: options.proxy,
      });

      return [translated1, translated2];
    }

    return [translated1];
  } else {
    return await Promise.all([
      simpleTranslate(text, {
        langFrom: options.langFrom,
        langTo: options.langTo,
        proxy: options.proxy,
      }),
      simpleTranslate(text, {
        langFrom: options.langTo[0],
        langTo: [options.langFrom],
        proxy: options.proxy,
      }),
    ]);
  }
}

export async function playTTS(text: string, langTo: string, proxy?: string) {
  const audioUrl = googleTTS.getAudioUrl(text, {
    lang: langTo,
    slow: false,
    host: "https://translate.google.com",
  });

  let agent: HttpsProxyAgent<string> | undefined;

  if (proxy) {
    try {
      agent = new HttpsProxyAgent(proxy);
    } catch (e) {
      console.error(`Error creating proxy agent for ${proxy}:`, e);
      agent = undefined; // Fallback to no proxy if agent creation fails
    }
  }

  // The options object for https.get. If 'agent' is undefined, it won't be included,
  // and https.get will use the default agent.
  const requestOptions: https.RequestOptions = {
    agent: agent,
  };

  https.get(audioUrl, requestOptions, (response) => {
    const chunks: Uint8Array[] = [];

    response.on("data", (chunk) => {
      chunks.push(chunk);
    });

    response
      .on("end", () => {
        const audioData = Buffer.concat(chunks);

        const tempFilePath = path.join(os.tmpdir(), "translation.mp3");
        writeFileSync(tempFilePath, audioData);

        // Play the audio file using afplay
        const afplayProcess = child_process.spawn("afplay", [tempFilePath]);

        afplayProcess.on("exit", (code) => {
          if (code !== 0) {
            console.error(`Error playing audio: afplay exited with code ${code}`);
          }
          if (existsSync(tempFilePath)) {
            unlinkSync(tempFilePath);
          }
        });
      })
      .on("error", (error) => {
        console.error("Error downloading audio:", error);
      });
  });
}
