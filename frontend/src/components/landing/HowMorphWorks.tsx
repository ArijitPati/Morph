"use client";

import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import {
  Phone,
  AudioLines,
  Cpu,
  Gauge,
  Bell,
} from "lucide-react";

const steps = [
  {
    num: "01",
    icon: Phone,
    title: "Voice Stream",
    description: "Incoming call audio is captured and routed to the analysis engine.",
  },
  {
    num: "02",
    icon: AudioLines,
    title: "Feature Extraction",
    description:
      "Acoustic, spectral, and prosodic characteristics are extracted from the voice signal.",
  },
  {
    num: "03",
    icon: Cpu,
    title: "AI Detection",
    description:
      "Morph's trained XGBoost classifier evaluates the extracted feature vector.",
  },
  {
    num: "04",
    icon: Gauge,
    title: "Risk Assessment",
    description:
      "Detection output is interpreted to produce a synthetic voice risk estimate.",
  },
  {
    num: "05",
    icon: Bell,
    title: "Alert",
    description:
      "Potential synthetic voice events are surfaced for security review.",
  },
];

export function HowMorphWorks() {
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
            How Morph Works
          </h2>
          <p className="mt-2 max-w-xl text-muted-foreground">
            The planned detection pipeline from voice capture to security alert.
          </p>
        </motion.div>

        <div className="relative">
          {/* Vertical connector line */}
          <div className="absolute left-6 top-0 bottom-0 hidden w-px bg-border lg:block" />

          <div className="space-y-6 lg:space-y-0 lg:grid lg:grid-cols-5 lg:gap-4">
            {steps.map((step, i) => (
              <motion.div
                key={step.num}
                initial={{ opacity: 0, y: 20 }}
                animate={isInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.4, delay: 0.12 * i }}
                className="relative flex gap-4 lg:flex-col lg:gap-0"
              >
                {/* Step number + icon */}
                <div className="relative z-10 flex-shrink-0">
                  <div className="flex h-12 w-12 items-center justify-center rounded-full border border-border bg-card">
                    <step.icon className="h-5 w-5 text-primary" />
                  </div>
                  <span className="absolute -top-2 -right-2 text-[10px] font-bold text-primary">
                    {step.num}
                  </span>
                </div>

                {/* Text */}
                <div className="pt-2 lg:pt-6 lg:text-center">
                  <h3 className="text-sm font-semibold">{step.title}</h3>
                  <p className="mt-1 text-sm text-muted-foreground lg:text-xs">
                    {step.description}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
