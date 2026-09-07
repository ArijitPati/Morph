"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { Phone, Upload } from "lucide-react";
import { VoiceVisualization } from "./VoiceVisualization";

export function Hero() {
  return (
    <section className="relative overflow-hidden pb-8 pt-20 sm:pt-28 lg:pt-32">
      {/* Background accent */}
      <div className="absolute inset-0 bg-gradient-to-b from-primary/[0.03] via-transparent to-transparent" />

      <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="grid gap-12 lg:grid-cols-2 lg:items-center lg:gap-16">
          {/* Text */}
          <div className="space-y-8">
            <motion.div
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="space-y-4"
            >
              <div className="inline-flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1">
                <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                <span className="text-xs font-medium text-muted-foreground">
                  SIH26104 — Voice Security
                </span>
              </div>
              <h1 className="text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl">
                Detect Synthetic Voices
                <br />
                <span className="text-primary">Before They Become Threats</span>
              </h1>
              <p className="max-w-lg text-lg text-muted-foreground">
                Morph analyzes incoming voice audio using acoustic and spectral
                characteristics to estimate whether a voice is genuine or
                potentially synthetic.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="flex flex-wrap gap-3"
            >
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
                Test a Recording
              </Link>
            </motion.div>
          </div>

          {/* Visualization */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.7, delay: 0.15 }}
            className="flex justify-center lg:justify-end"
          >
            <VoiceVisualization />
          </motion.div>
        </div>
      </div>
    </section>
  );
}
