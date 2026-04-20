'use client';

import { motion } from 'framer-motion';
import type { FeaturedGroup, FeaturedItem, SourceFilter, Suggestion } from '@/lib/types';

type SearchProps = {
    mode: 'search';
    suggestions: Suggestion[];
    keyword: string;
    activeIndex: number;
    sourceFilter: SourceFilter;
    isFallback: boolean;
    isLoading: boolean;
    onHover: (index: number) => void;
    onSelectSuggestion: (s: Suggestion) => void;
    onSourceChange: (filter: SourceFilter) => void;
};

type FeaturedProps = {
    mode: 'featured';
    groups: FeaturedGroup[];
    filteredItems: FeaturedItem[];
    activeCategory: string | null;
    activeIndex: number;
    onHover: (index: number) => void;
    onCategoryChange: (intent: string | null) => void;
    onSelectFeatured: (item: FeaturedItem) => void;
};

type SuggestBoxProps = SearchProps | FeaturedProps;

// Wrap every occurrence of the keyword in an emerald-bold span. Match case-insensitively.
function highlightMatch(text: string, keyword: string): React.ReactNode {
    const kw = keyword.trim().toLowerCase();
    if (!kw) return text;

    const lower = text.toLowerCase();
    const parts: React.ReactNode[] = [];
    let cursor = 0;
    let idx = lower.indexOf(kw, cursor);
    if (idx < 0) return text;

    while (idx >= 0) {
        if (idx > cursor) parts.push(text.slice(cursor, idx));
        parts.push(
            <span key={`m-${idx}`} className="font-semibold text-emerald-700">
                {text.slice(idx, idx + kw.length)}
            </span>,
        );
        cursor = idx + kw.length;
        idx = lower.indexOf(kw, cursor);
    }
    if (cursor < text.length) parts.push(text.slice(cursor));
    return <>{parts}</>;
}

// Map source_type to a muted chip palette — amber for PDFs, sky for websites, slate for mixed.
const SOURCE_STYLES: Record<string, string> = {
    pdf: 'bg-amber-50 text-amber-700 ring-amber-600/10',
    website: 'bg-sky-50 text-sky-700 ring-sky-600/10',
    mixed: 'bg-slate-100 text-slate-500 ring-slate-600/10',
};

function SourceChip({ sourceType }: { sourceType: string }) {
    const style = SOURCE_STYLES[sourceType] ?? SOURCE_STYLES.mixed;
    return (
        <span
            className={`inline-flex shrink-0 items-center rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider ring-1 ring-inset ${style}`}
        >
            {sourceType}
        </span>
    );
}

// Map intent to its own chip palette — kept distinct from source colors so badges read at a glance.
const INTENT_STYLES: Record<string, string> = {
    general_overview: 'bg-emerald-50 text-emerald-700 ring-emerald-600/15',
    project_initiative: 'bg-indigo-50 text-indigo-700 ring-indigo-600/15',
    publication_finding: 'bg-teal-50 text-teal-700 ring-teal-600/15',
    topic_specific: 'bg-rose-50 text-rose-700 ring-rose-600/15',
    synthesis: 'bg-violet-50 text-violet-700 ring-violet-600/15',
};

const INTENT_SHORT: Record<string, string> = {
    general_overview: 'Overview',
    project_initiative: 'Project',
    publication_finding: 'Publication',
    topic_specific: 'Topic',
    synthesis: 'Synthesis',
};

function IntentBadge({ intent }: { intent: string }) {
    const style = INTENT_STYLES[intent] ?? 'bg-slate-100 text-slate-600 ring-slate-600/10';
    const label = INTENT_SHORT[intent] ?? intent.replace('_', ' ');
    return (
        <span
            className={`inline-flex shrink-0 items-center rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider ring-1 ring-inset ${style}`}
        >
            {label}
        </span>
    );
}

// Pill button used in the featured-mode category filter row.
function CategoryPill({
    label,
    active,
    onClick,
}: {
    label: string;
    active: boolean;
    onClick: () => void;
}) {
    return (
        <button
            type="button"
            // Suppress textarea blur — same trick used by suggestion rows.
            onMouseDown={(e) => {
                e.preventDefault();
                onClick();
            }}
            className={`cursor-pointer rounded-full px-3 py-1 text-xs font-medium ring-1 ring-inset transition-colors ${
                active
                    ? 'bg-emerald-50 text-emerald-700 ring-emerald-600/30'
                    : 'bg-slate-50 text-slate-600 ring-slate-200 hover:bg-slate-100'
            }`}
        >
            {label}
        </button>
    );
}

// Render a placeholder row while the fetch is in flight — keeps the dropdown footprint stable.
function SkeletonRow({ index }: { index: number }) {
    // Stagger row widths so the pulse reads as a list, not a single uniform shape.
    const widths = ['w-3/4', 'w-2/3', 'w-4/5'];
    return (
        <li className="px-4 py-3" aria-hidden="true">
            <div className="flex items-center gap-3">
                <div
                    className={`h-3 animate-pulse rounded bg-slate-200/70 ${widths[index % widths.length]}`}
                />
                <div className="ml-auto h-4 w-12 shrink-0 animate-pulse rounded bg-slate-100" />
            </div>
        </li>
    );
}

// Stagger entrance — items cascade in once on mount of the variant tree.
const LIST_VARIANTS = {
    hidden: {},
    visible: { transition: { staggerChildren: 0.035, delayChildren: 0.04 } },
};

const ITEM_VARIANTS = {
    hidden: { opacity: 0, y: 6 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.22, ease: 'easeOut' } },
};

// Shared shell — both modes render inside the same dropdown frame for visual continuity.
function Shell({ children, label }: { children: React.ReactNode; label: string }) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 8, scale: 0.985 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.985 }}
            transition={{ type: 'spring', stiffness: 260, damping: 28 }}
            className="absolute bottom-full left-0 right-0 mb-2 overflow-hidden rounded-xl border border-slate-200/60 bg-white shadow-[0_12px_40px_-8px_rgba(15,23,42,0.12),0_4px_12px_-4px_rgba(15,23,42,0.05)]"
            role="listbox"
            aria-label={label}
        >
            {children}
        </motion.div>
    );
}

export default function SuggestBox(props: SuggestBoxProps) {
    if (props.mode === 'search') {
        // Bail only when there's nothing to show AND nothing in flight — otherwise loading skeletons fill the space.
        if (props.suggestions.length === 0 && !props.isLoading) return null;

        const headerLabel = props.isFallback ? 'Did you mean?' : 'Matches';

        return (
            <Shell label={headerLabel}>
                {/* Header strip — fallback variant softens the prompt; strict pass shows source filter pills. */}
                <div className="border-b border-slate-100 bg-linear-to-b from-slate-50/80 to-white px-4 pb-3 pt-3">
                    <div className="text-sm font-semibold text-slate-800">{headerLabel}</div>
                    {props.isFallback ? (
                        <div className="mt-1 text-[11px] text-slate-500">
                            No exact match — closest topics below.
                        </div>
                    ) : (
                        <div className="mt-3 flex flex-wrap gap-1.5">
                            <CategoryPill
                                label="All"
                                active={props.sourceFilter === null}
                                onClick={() => props.onSourceChange(null)}
                            />
                            <CategoryPill
                                label="PDFs"
                                active={props.sourceFilter === 'pdf'}
                                onClick={() => props.onSourceChange('pdf')}
                            />
                            <CategoryPill
                                label="Websites"
                                active={props.sourceFilter === 'website'}
                                onClick={() => props.onSourceChange('website')}
                            />
                        </div>
                    )}
                </div>

                <motion.ul
                    // Re-key on filter / fallback flip to retrigger the stagger entrance — loading flips don't restart the animation.
                    key={`search-${props.sourceFilter ?? 'all'}-${props.isFallback ? 'fb' : 'ok'}`}
                    variants={LIST_VARIANTS}
                    initial="hidden"
                    animate="visible"
                    className="max-h-80 overflow-y-auto py-0"
                >
                    {props.isLoading && props.suggestions.length === 0
                        ? [0, 1, 2].map((i) => <SkeletonRow key={`sk-${i}`} index={i} />)
                        : null}
                    {!props.isLoading &&
                        props.suggestions.map((s, i) => {
                            const isActive = i === props.activeIndex;
                            return (
                                <motion.li
                                    key={s.qa_id}
                                    variants={ITEM_VARIANTS}
                                    role="option"
                                    aria-selected={isActive}
                                    onMouseEnter={() => props.onHover(i)}
                                    onMouseDown={(e) => {
                                        e.preventDefault();
                                        props.onSelectSuggestion(s);
                                    }}
                                    className={`relative flex cursor-pointer items-center gap-3 px-4 py-3 text-sm transition-colors ${
                                        isActive
                                            ? 'bg-emerald-50/70 text-emerald-950 shadow-[inset_3px_0_0_0_rgb(16,185,129)]'
                                            : props.isFallback
                                              ? 'text-slate-500 hover:bg-slate-50'
                                              : 'text-slate-700 hover:bg-slate-50'
                                    }`}
                                >
                                    <span className="flex-1 truncate leading-snug">
                                        {props.isFallback
                                            ? s.question
                                            : highlightMatch(s.question, props.keyword)}
                                    </span>
                                    <SourceChip sourceType={s.source_type} />
                                </motion.li>
                            );
                        })}
                </motion.ul>
            </Shell>
        );
    }

    // Featured mode — header strip, category pills, intent-badged rows, optional spotlight first item.
    if (props.filteredItems.length === 0) return null;

    const showSpotlight = props.activeCategory === null;

    return (
        <Shell label="Featured questions">
            {/* Header strip — sets the context so the dropdown reads as curated, not search results. */}
            <div className="border-b border-slate-100 bg-linear-to-b from-slate-50/80 to-white px-4 pb-3 pt-3">
                <div className="flex items-baseline justify-between gap-2">
                    <div>
                        <div className="text-sm font-semibold text-slate-800">
                            Suggested questions
                        </div>
                    </div>
                </div>

                {/* Category filter — "All" plus one pill per group present in the response. */}
                <div className="mt-3 flex flex-wrap gap-1.5">
                    <CategoryPill
                        label="All"
                        active={props.activeCategory === null}
                        onClick={() => props.onCategoryChange(null)}
                    />
                    {props.groups.map((g) => (
                        <CategoryPill
                            key={g.intent}
                            label={g.label}
                            active={props.activeCategory === g.intent}
                            onClick={() => props.onCategoryChange(g.intent)}
                        />
                    ))}
                </div>
            </div>

            <motion.ul
                // Re-key on category change to retrigger the stagger entrance for the new list.
                key={`featured-${props.activeCategory ?? 'all'}`}
                variants={LIST_VARIANTS}
                initial="hidden"
                animate="visible"
                className="max-h-80 overflow-y-auto py-1"
            >
                {props.filteredItems.map((item, i) => {
                    const isActive = i === props.activeIndex;
                    const isSpotlight = showSpotlight && i === 0;
                    return (
                        <motion.li
                            key={item.qa_id}
                            variants={ITEM_VARIANTS}
                            role="option"
                            aria-selected={isActive}
                            onMouseEnter={() => props.onHover(i)}
                            onMouseDown={(e) => {
                                e.preventDefault();
                                props.onSelectFeatured(item);
                            }}
                            className={`relative cursor-pointer px-4 transition-colors ${
                                isSpotlight ? 'py-3.5' : 'py-3'
                            } ${
                                isActive
                                    ? 'bg-emerald-50/70 text-emerald-950 shadow-[inset_3px_0_0_0_rgb(16,185,129)]'
                                    : 'text-slate-700 hover:bg-slate-50'
                            }`}
                        >
                            {isSpotlight && (
                                <div className="mb-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-emerald-600">
                                    Featured
                                </div>
                            )}
                            <div className="flex items-center gap-3">
                                <span
                                    className={`flex-1 truncate leading-snug ${isSpotlight ? 'text-[15px] font-medium' : 'text-sm'}`}
                                >
                                    {item.question}
                                </span>
                                <IntentBadge intent={item.intent} />
                                <SourceChip sourceType={item.source_type} />
                            </div>
                        </motion.li>
                    );
                })}
            </motion.ul>
        </Shell>
    );
}
