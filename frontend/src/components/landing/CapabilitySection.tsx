"use client";

import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import { Radio, AudioLines, ShieldAlert, Lock } from "lucide-react";

const capabilities = [
  {
    icon: Radio,
    title: "Real-Time Voice Analysis",
    description:
      "Analyze incoming voice streams during communication to identify potential synthetic speech in real time.",
  },
  {
    icon: AudioLines,
    title: "Acoustic & Spectral Analysis",
    description:
      "Examine 132 voice characteristics — MFCC, CQCC, spectral features, and pitch — used by the Morph detection engine.",
  },
  {
    icon: ShieldAlert,
    title: "Synthetic Voice Risk Assessment",
    description:
      "Estimate the likelihood that a voice is AI-generated or cloned using trained XGBoost classifiers.",
  },
  {
    icon: Lock,
    title: "Privacy-Aware Processing",
    description:
      "Analyze acoustic signatures without requiring permanent storage of raw conversational content.",
  },
];

export function CapabilitySection() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section ref={ref} className="py-20">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5 }}
          className="mb-12"
        >
          <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">
            Built for Voice Security
          </h2>
          <p className="mt-2 max-w-xl text-muted-foreground">
            Core capabilities designed to protect communications from synthetic
            voice threats.
          </p>
        </motion.div>

        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {capabilities.map((cap, i) => (
            <motion.div
              key={cap.title}
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.4, delay: 0.1 * i }}
              className="rounded-xl border border-border bg-card p-6 transition-colors hover:border-primary/20"
            >
              <cap.icon className="h-5 w-5 text-primary" />
              <h3 className="mt-4 text-sm font-semibold">{cap.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                {cap.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
