import "./App.css";
import Autocomplete from "./Autocomplete";
import { Button } from "./components/ui/button";
import { movies } from "./movies";
import { findNearestMovie, type NeighborMovie } from "./helpers/findNearestMovie";
import { useEffect, useState } from "react";
import { loadEmbeddings } from "./helpers/loadEmbeddings";
import { idToTitle } from "./helpers/idToTitle";
import { Film, Loader2, Popcorn, ArrowRight, Settings2 } from "lucide-react";

export default function App() {
    const [currentInput, setCurrentInput] = useState<string>("");
    const [results, setResults] = useState<NeighborMovie[]>([]);
    const [loading, setLoading] = useState(true);
    const [isSearching, setIsSearching] = useState(false);
    const [topK, setTopK] = useState(10);

    const [titles] = useState<string[]>(() => movies.map((m) => m[1] as string));
    const [placeholder] = useState<string>(() => `e.g., ${titles[Math.floor(Math.random() * titles.length)]}`);

    useEffect(() => {
        loadEmbeddings().then(() => setLoading(false)).catch((e) => {
            console.error(e);
            setLoading(false);
        });
    }, []);

    const handleSearch = () => {
        if (!currentInput) return;
        setIsSearching(true);
        setTimeout(() => {
            try {
                const currentResults = findNearestMovie(currentInput, topK);
                setResults(currentResults);
            } catch (error) {
                console.error(error);
            } finally {
                setIsSearching(false);
            }
        }, 300);
    };

    const handleTopKChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        let val = parseInt(e.target.value);
        if (isNaN(val)) val = 1;
        if (val > 20) val = 20;
        if (val < 1) val = 1;
        setTopK(val);
    };

    return (
        <div className="min-h-screen w-full bg-stone-950 text-stone-200 relative selection:bg-stone-500/30">
            <div className="relative z-10 container mx-auto px-4 pt-12 pb-20 flex flex-col items-center max-w-5xl">
                <div className="text-center mb-12">
                    <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-stone-200">
                        Find your next favorite movie
                    </h1>
                </div>

                <div className="w-full max-w-2xl space-y-6">
                    <div className="bg-stone-900/40 p-1 rounded-3xl border border-stone-800/60 shadow-2xl backdrop-blur-sm">
                        <div className="bg-stone-950/50 p-6 md:p-8 rounded-[1.3rem] space-y-6">
                            <div className="space-y-3">
                                <label className="text-sm font-medium text-stone-400 ml-1 flex items-center gap-2">
                                    <Film className="w-4 h-4" />
                                    What did you watch recently?
                                </label>
                                <Autocomplete
                                    value={currentInput}
                                    setValue={setCurrentInput}
                                    items={titles}
                                    placeholder={placeholder}
                                />
                            </div>

                            <div className="space-y-3">
                                <div className="flex items-center justify-between ml-1">
                                    <label className="text-sm font-medium text-stone-400 flex items-center gap-2">
                                        <Settings2 className="w-4 h-4" />
                                        Results count
                                    </label>
                                </div>
                                <div className="relative">
                                    <input 
                                        type="number"
                                        min="1" 
                                        max="20" 
                                        value={topK} 
                                        onChange={handleTopKChange}
                                        className="w-full bg-stone-900/50 border border-stone-800 text-stone-200 text-sm rounded-xl focus:border-stone-600 focus:ring-2 focus:ring-white/5 block p-3 transition-all outline-none"
                                    />
                                    <div className="absolute inset-y-0 right-0 flex items-center pr-4 pointer-events-none text-stone-600 text-xs font-mono">
                                        MAX 20
                                    </div>
                                </div>
                            </div>
                            
                            <Button
                                className="w-full h-14 text-lg font-medium bg-stone-300 text-stone-900 hover:bg-stone-200 rounded-xl transition-all duration-300 shadow-xl shadow-black/20 cursor-pointer"
                                onClick={handleSearch}
                                disabled={!currentInput || isSearching || loading}
                            >
                                {isSearching || loading ? (
                                    <>
                                        <Loader2 className="w-5 h-5 animate-spin mr-2" />
                                        {loading ? "Loading Model..." : "Finding matches..."}
                                    </>
                                ) : (
                                    <>
                                        <Popcorn className="w-5 h-5 mr-2" />
                                        Find Similar Movies
                                    </>
                                )}
                            </Button>
                        </div>
                    </div>
                </div>

                {results.length > 0 && (
                    <div className="w-full mt-16">
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6">
                            {results.map((result, index) => (
                                <a 
                                    key={result.movieID}
                                    href={`https://movielens.org/movies/${result.movieID}`} 
                                    target="_blank"
                                    rel="noreferrer"
                                    className="group relative flex flex-col bg-stone-900/40 hover:bg-stone-900 border border-stone-800 hover:border-stone-600 rounded-2xl p-5 transition-all duration-300"
                                >
                                    <div className="flex items-start justify-between mb-4 relative z-10">
                                        <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-stone-800 border border-stone-700 text-stone-400 font-bold font-mono text-lg group-hover:bg-stone-200 group-hover:text-stone-900 group-hover:border-stone-400 transition-all duration-300 shadow-sm">
                                            {index + 1}
                                        </div>
                                        <div className="p-2 rounded-full text-stone-600 group-hover:text-stone-300 transition-colors">
                                            <ArrowRight className="w-4 h-4" />
                                        </div>
                                    </div>
                                    
                                    <div className="mt-auto relative z-10">
                                        <h3 className="font-semibold text-lg text-stone-300 group-hover:text-stone-100 leading-tight transition-colors mb-1 line-clamp-2">
                                            {idToTitle(result.movieID)}
                                        </h3>
                                        <p className="text-sm text-stone-500 group-hover:text-stone-400 transition-colors">
                                            Similarity: <span className="font-mono">{(result.score).toFixed(4)}</span>
                                        </p>
                                    </div>
                                </a>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
