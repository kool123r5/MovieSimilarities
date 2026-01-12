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
        <div className="w-full max-w-2xl mx-auto p-6">
            <div className="relative">
                <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-white/40" />
                    <Input
                        ref={inputRef}
                        value={value}
                        placeholder={placeholder}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setValue(e.target.value)}
                        onKeyDown={onKeyDown}
                        onFocus={() => filtered.length && setOpen(true)}
                        className="pl-10 h-12 bg-white/5 border-white/10 text-white placeholder:text-white/30 focus:border-white/20 focus:ring-2 focus:ring-white/10"
                        aria-autocomplete="list"
                        aria-expanded={open}
                    />
                </div>

                {open && filtered.length > 0 && (
                    <div
                        ref={listRef}
                        role="listbox"
                        className="absolute left-0 right-0 mt-2 max-h-80 overflow-auto rounded-lg border border-white/10 bg-zinc-900/95 backdrop-blur-sm shadow-2xl z-50"
                    >
                        <div className="p-1.5">
                            {filtered.map((item, index) => (
                                <div
                                    key={`${item}-${index}`}
                                    data-item
                                    role="option"
                                    aria-selected={index === highlight}
                                    className={`
                                            px-3 py-2.5 rounded-md cursor-pointer transition-all duration-75
                                            ${
                                                index === highlight
                                                    ? "bg-white text-zinc-900 font-medium"
                                                    : "text-white/90 hover:bg-white/10"
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
        </div>
    );
}
