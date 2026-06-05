"use client";

type ModeToggleProps = {
  mode: "simple" | "technical";
  onChange: (mode: "simple" | "technical") => void;
};

export default function ModeToggle({ mode, onChange }: ModeToggleProps) {
  return (
    <div className="modeToggle" aria-label="Result mode">
      <button className={mode === "simple" ? "active" : ""} onClick={() => onChange("simple")} type="button">
        Simple
      </button>
      <button className={mode === "technical" ? "active" : ""} onClick={() => onChange("technical")} type="button">
        Technical
      </button>
    </div>
  );
}
