import Link from "next/link";

export default function StoryPage() {
  return (
    <main className="storyPage">
      <Link className="backLink" href="/">Back to search</Link>
      <section className="storyHero">
        <p className="kicker">The Story Behind CORTEX</p>
        <h1>From retrieval experiments to governed route-aware ranking.</h1>
        <p>
          CORTEX began with a simple question: what if product search treated intent, constraints, and governance as first-class ranking signals?
        </p>
      </section>

      <section className="storyContent">
        <article>
          <h2>What problem CORTEX solves</h2>
          <p>
            Commerce search is not a single task. A shopper may ask for a product, a setup, a gift, a replacement, or a constraint-heavy item. CORTEX routes these queries differently so retrieval behavior matches intent.
          </p>
        </article>
        <article>
          <h2>Why search is not just retrieval</h2>
          <p>
            Retrieval finds candidates. Governed ranking decides which retrieval path should be trusted, repaired, constrained, reviewed, or preserved.
          </p>
        </article>
        <article>
          <h2>Scale introduces near-matches</h2>
          <p>
            Larger indexes improve coverage, but they can also introduce noisy lexical neighbors. The 500k comparison showed that more candidates do not automatically produce better route-level quality.
          </p>
        </article>
        <article>
          <h2>Governed route-aware ranking</h2>
          <p>
            CORTEX coordinates baseline preservation, strict repair, mission repair, critic review, behavior-aware reranking, fallback control, and final slate building as one system.
          </p>
        </article>
        <article>
          <h2>Strict constraints</h2>
          <p>
            Negation and compatibility constraints are tracked with clean, removed, demoted, hard violation, and soft violation diagnostics. This makes failures measurable instead of invisible.
          </p>
        </article>
        <article>
          <h2>Scale progression</h2>
          <p>
            The local FTS backend progressed from 100k to 500k to 1M ESCI examples, reaching 760,149 products and 50,225 queries in the 1M validation index.
          </p>
        </article>
        <article>
          <h2>The negative result</h2>
          <p>
            strict_boost was active, but did not recover quality. This is the key research signal: simple penalty reranking is not enough.
          </p>
        </article>
        <article>
          <h2>What comes next</h2>
          <p>
            The product showcase can add route traces, slate comparison, and high-touch demos while Track A remains a reproducible research backend.
          </p>
        </article>
      </section>
    </main>
  );
}
