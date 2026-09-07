import { Shield } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-border bg-background/50 mt-auto">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Shield className="h-3.5 w-3.5" />
          <span>Morph Voice Detection System</span>
          <span className="text-border">|</span>
          <span>SIH26104</span>
        </div>
        <div className="text-xs text-muted-foreground">
          Secure Communications Division
        </div>
      </div>
    </footer>
  );
}
