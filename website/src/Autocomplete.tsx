import { useEffect, useRef, useState } from "react";
import { Input } from "@/components/ui/input";
import { Search } from "lucide-react";

interface AutocompleteProps {
    items: string[];
    placeholder?: string;
    onSelect?: (value: string) => void;
    debounceMs?: number;
    value: string;
    setValue: (value: string) => void;
}

export default function Autocomplete({
    items,
    placeholder = "Search for a movie...",
    onSelect = () => {},
    debounceMs = 150,
    value,
    setValue,
}: AutocompleteProps) {
    const [filtered, setFiltered] = useState<string[]>([]);
    const [open, setOpen] = useState<boolean>(false);
    const [highlight, setHighlight] = useState<number>(-1);

    const inputRef = useRef<HTMLInputElement | null>(null);
    const listRef = useRef<HTMLDivElement | null>(null);
    const timerRef = useRef<number | null>(null);

    useEffect(() => {
        if (timerRef.current) window.clearTimeout(timerRef.current);

        timerRef.current = window.setTimeout(() => {
            const q = value.trim();
            if (!q) {
                setFiltered([]);
                setOpen(false);
                setHighlight(-1);
                return;
            }

            const query = q.toLowerCase();

            const scored = items
                .map((item) => {
                    const lower = item.toLowerCase();
                    let score = 0;

                    if (lower === query) {
                        score = 0;
                    } else if (lower.startsWith(query)) {
                        score = 500;
                    } else if (lower.includes(` ${query}`)) {
                        score = 250;
                    } else if (lower.includes(query)) {
                        score = 100;
                    }

                    return { item, score };
                })
                .filter((x) => x.score > 0)
                .sort((a, b) => b.score - a.score)
                .slice(0, 12)
                .map((x) => x.item);

            setFiltered(scored);
            setOpen(scored.length > 0);
            setHighlight(scored.length > 0 ? 0 : -1);
        }, debounceMs);

        return () => {
            if (timerRef.current) window.clearTimeout(timerRef.current);
        };
    }, [value, items, debounceMs]);

    function pick(item: string) {
        setOpen(false);
        setValue(item);
        setFiltered([]);
        setHighlight(-1);
        onSelect(item);
    }

    function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
        if (e.key === "ArrowDown") {
            e.preventDefault();
            setHighlight((h) => (filtered.length ? (h + 1) % filtered.length : 0));
        }

        if (e.key === "ArrowUp") {
            e.preventDefault();
            setHighlight((h) => (filtered.length ? (h <= 0 ? filtered.length - 1 : h - 1) : -1));
        }

        if (e.key === "Enter" && open && highlight >= 0 && highlight < filtered.length) {
            e.preventDefault();
            pick(filtered[highlight]);
        }

        if (e.key === "Escape") {
            setOpen(false);
        }
    }

    return (
        <div className="w-full relative group">
            <div className="relative transition-all duration-300 ease-in-out">
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-stone-500 group-focus-within:text-stone-300 transition-colors" />
                <Input
                    ref={inputRef}
                    value={value}
                    placeholder={placeholder}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => setValue(e.target.value)}
                    onKeyDown={onKeyDown}
                    onFocus={() => filtered.length && setOpen(true)}
                    className="pl-12 h-14 bg-stone-900/50 border-stone-800 text-stone-100 placeholder:text-stone-600 focus:border-stone-600 focus:ring-4 focus:ring-white/5 rounded-2xl shadow-sm text-base transition-all"
                    aria-autocomplete="list"
                    aria-expanded={open}
                />
            </div>

            {open && filtered.length > 0 && (
                <div
                    ref={listRef}
                    role="listbox"
                    className="absolute left-0 right-0 mt-2 max-h-80 overflow-auto rounded-xl border border-stone-800 bg-stone-900/95 backdrop-blur-xl shadow-2xl z-50 scrollbar-thin scrollbar-thumb-stone-700 scrollbar-track-transparent"
                >
                    <div className="p-2 space-y-0.5">
                        {filtered.map((item, index) => (
                            <div
                                key={`${item}-${index}`}
                                data-item
                                role="option"
                                aria-selected={index === highlight}
                                className={`
                                        px-4 py-3 rounded-lg cursor-pointer text-sm transition-all duration-150 flex items-center
                                        ${
                                            index === highlight
                                                ? "bg-stone-100 text-stone-900 shadow-md shadow-black/20"
                                                : "text-stone-300 hover:bg-stone-800/50 hover:text-white"
                                        }
                                    `}
                                onMouseEnter={() => setHighlight(index)}
                                onMouseDown={(e) => {
                                    e.preventDefault();
                                    pick(item);
                                }}
                            >
                                {item}
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
