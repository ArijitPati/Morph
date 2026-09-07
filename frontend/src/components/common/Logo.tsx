"use client";

import { Shield } from "lucide-react";
import Link from "next/link";

export function Logo() {
  return (
    <Link href="/" className="flex items-center gap-2.5 group">
      <div className="relative rounded-lg bg-primary/10 p-1.5 transition-colors group-hover:bg-primary/20">
        <Shield className="h-5 w-5 text-primary" />
        <div className="absolute inset-0 rounded-lg bg-primary/5 blur-sm" />
      </div>
      <div className="flex flex-col">
        <span className="text-sm font-bold tracking-tight leading-none">
          MORPH
        </span>
        <span className="text-[10px] font-medium text-muted-foreground tracking-widest uppercase">
          Voice Guard
        </span>
      </div>
    </Link>
  );
}
