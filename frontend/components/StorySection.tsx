import Link from "next/link";

export default function StorySection() {
  return (
    <section className="storyBand">
      <div>
        <p className="kicker">Research system</p>
        <h2>The Story Behind CORTEX</h2>
        <p>From retrieval experiments to governed route-aware ranking.</p>
      </div>
      <Link href="/story">Read the full story</Link>
    </section>
  );
}
