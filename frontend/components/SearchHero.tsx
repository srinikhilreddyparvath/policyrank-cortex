"use client";

import { FormEvent } from "react";

type SearchHeroProps = {
  query: string;
  loading: boolean;
  onQueryChange: (query: string) => void;
  onSearch: () => void;
};

export default function SearchHero({ query, loading, onQueryChange, onSearch }: SearchHeroProps) {
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSearch();
  }

  return (
    <section className="hero">
      <div className="heroAura" />
      <div className="heroInner">
        <p className="kicker">Governed product search</p>
        <h1>CORTEX Search Intelligence</h1>
        <p className="subtitle">Search at scale. Route with intent. Rank with governance.</p>
        <form className="searchForm" onSubmit={submit}>
          <input
            aria-label="Search query"
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            spellCheck={false}
          />
          <button disabled={loading || !query.trim()} type="submit">
            {loading ? "Searching" : "Search"}
          </button>
        </form>
      </div>
    </section>
  );
}
