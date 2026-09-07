"use client";

import { motion, useInView } from "framer-motion";
import { useRef } from "react";

export function SecuritySection() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section ref={ref} className="py-20">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5 }}
          className="relative overflow-hidden rounded-xl border border-border bg-card p-8 sm:p-12"
        >
          {/* Background waveform pattern */}
          <div className="absolute inset-0 overflow-hidden opacity-[0.04]">
            <svg
              className="h-full w-full"
              viewBox="0 0 800 200"
              preserveAspectRatio="none"
            >
              {Array.from({ length: 80 }, (_, i) => {
                const x = i * 10;
                const h = 10 + Math.sin(i * 0.4) * 30 + Math.sin(i * 0.7) * 20;
                return (
                  <rect
                    key={i}
                    x={x}
                    y={100 - h / 2}
                    width={4}
                    height={h}
                    rx={2}
                    fill="currentColor"
                    className="text-primary"
                  />
                );
              })}
            </svg>
          </div>

          <div className="relative grid gap-8 lg:grid-cols-2 lg:items-center">
            <div className="space-y-4">
              <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">
                Every voice carries a signal.
                <br />
                <span className="text-primary">
                  Morph looks beyond the words.
                </span>
              </h2>
              <p className="max-w-md text-muted-foreground">
                Morph focuses on acoustic authenticity characteristics — the
                spectral fingerprint, prosodic patterns, and cepstral signatures
                that distinguish genuine human speech from synthetic generation.
              </p>
            </div>

            {/* Mini frequency bars */}
            <div className="flex items-end justify-center gap-[3px] h-32 lg:h-40">
              {Array.from({ length: 40 }, (_, i) => {
                const base = Math.sin(i * 0.25) * 0.4 + 0.5;
                const noise = Math.sin(i * 0.6) * 0.2;
                const height = Math.max(0.08, base + noise);
                return (
                  <motion.div
                    key={i}
                    initial={{ scaleY: 0 }}
                    animate={isInView ? { scaleY: 1 } : {}}
                    transition={{
                      duration: 0.5,
                      delay: 0.02 * i,
                      ease: "easeOut",
                    }}
                    className="w-1.5 rounded-full bg-primary/20 origin-bottom"
                    style={{ height: `${height * 100}%` }}
                  />
                );
              })}
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
