"use client";

import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import Link from "next/link";
import { Phone, Upload } from "lucide-react";

export function FinalCTA() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section ref={ref} className="py-20">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5 }}
          className="relative overflow-hidden rounded-xl border border-border bg-card p-8 text-center sm:p-12"
        >
          <div className="absolute inset-0 bg-gradient-to-b from-primary/[0.04] via-transparent to-transparent" />

          <div className="relative space-y-4">
            <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">
              Ready to inspect the voice?
            </h2>
            <p className="mx-auto max-w-md text-muted-foreground">
              Test Morph with a recording or explore the live-call detection
              workflow.
            </p>
            <div className="flex items-center justify-center gap-3 pt-4">
              <Link
                href="/call"
                className="inline-flex items-center gap-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground transition-colors hover:bg-primary/90"
              >
                <Phone className="h-4 w-4" />
                Start Live Call
              </Link>
              <Link
                href="/detection"
                className="inline-flex items-center gap-2 rounded-lg border border-border bg-card px-5 py-2.5 text-sm font-semibold transition-colors hover:bg-secondary"
              >
                <Upload className="h-4 w-4" />
                Analyze Recording
              </Link>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
