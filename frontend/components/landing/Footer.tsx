"use client";

export default function Footer() {
  return (
    <footer className="border-t border-slate-200/50 py-8">
      <div className="mx-auto flex max-w-[1400px] items-center justify-between px-6">
        <p className="text-sm text-slate-400">
          Sustainable Solutions Lab &middot; {new Date().getFullYear()}
        </p>
        <a
          href="#"
          className="text-sm text-slate-400 transition-colors hover:text-zinc-900"
          onClick={(e) => {
            e.preventDefault();
            window.scrollTo({ top: 0, behavior: "smooth" });
          }}
        >
          Back to top
        </a>
      </div>
    </footer>
  );
}
