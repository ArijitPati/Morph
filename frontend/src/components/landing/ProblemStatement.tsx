"use client";

import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import { AlertTriangle } from "lucide-react";

export function ProblemStatement() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section ref={ref} className="py-20">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5 }}
          className="rounded-xl border border-border bg-card p-8 sm:p-10"
        >
          <div className="flex items-start gap-4">
            <div className="rounded-lg bg-warning/10 p-2.5">
              <AlertTriangle className="h-5 w-5 text-warning" />
            </div>
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold tracking-tight">
                  SIH26104
                </h2>
                <span className="rounded-md bg-secondary px-2 py-0.5 text-xs font-medium text-muted-foreground">
                  Problem Statement
                </span>
              </div>
              <h3 className="text-xl font-semibold leading-snug">
                AI-Powered Real-Time Detection &amp; Prevention of Voice Cloning
                Impersonation Attacks
              </h3>
              <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground">
                Voice cloning technology can make an attacker sound like a
                trusted person — a colleague, executive, or family member. Morph
                is designed to analyze the incoming voice signal and identify
                signals associated with synthetic or cloned speech, providing an
                additional layer of verification for voice-based communications.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
