import type { ReactNode } from "react";

// Wraps the dense "real, disclosed limitation" methodology prose this
// project's cards are built around -- collapsed by default so the
// default view reads like a product, not a paper, while keeping every
// word of the disclosure one click away, never deleted or shortened.
export default function Disclosure({
  summary, children, defaultOpen = false,
}: { summary: string; children: ReactNode; defaultOpen?: boolean }) {
  return (
    <details className="disclosure" open={defaultOpen}>
      <summary>{summary}</summary>
      <div className="disclosure-body">{children}</div>
    </details>
  );
}
