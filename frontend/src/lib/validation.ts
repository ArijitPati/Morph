import { z } from "zod";

export const AudioFileSchema = z.object({
  file: z
    .instanceof(File)
    .refine(
      (f) => ["audio/wav", "audio/flac", "audio/x-wav"].includes(f.type) || 
             [".wav", ".flac"].some(ext => f.name.toLowerCase().endsWith(ext)),
      "Only WAV and FLAC files are supported"
    )
    .refine((f) => f.size <= 50 * 1024 * 1024, "File size must be under 50MB"),
});

export const ModelVersionSchema = z.enum(["v1", "v2"]);

export const DetectionConfigSchema = z.object({
  modelVersion: ModelVersionSchema.default("v2"),
});

export type AudioFileInput = z.infer<typeof AudioFileSchema>;
export type ModelVersion = z.infer<typeof ModelVersionSchema>;
