import "./App.css";
import Autocomplete from "./Autocomplete";
import { Button } from "./components/ui/button";
import { movies } from "./movies";
import { findNearestMovie, type NeighborMovie } from "./helpers/findNearestMovie";
import { useEffect, useState } from "react";
import { loadEmbeddings } from "./helpers/loadEmbeddings";
import { idToTitle } from "./helpers/idToTitle";

export default function App() {
    const [currentInput, setCurrentInput] = useState<string>("");
    const [results, setResults] = useState<NeighborMovie[]>([]);
    let titles: string[] = [];
    for (let i = 0; i < movies.length; i++) {
        const movie = movies[i];
        titles.push(movie[1] as string);
    }
    useEffect(() => {
        loadEmbeddings();
    }, []);
    return (
        <div className="App">
            <div className="flex justify-center flex-col w-full items-center">
                <h1>Enter a movie you *currently* like</h1>

                <Autocomplete
                    value={currentInput}
                    setValue={setCurrentInput}
                    items={titles}
                    placeholder={`Eg: ${titles[Math.floor(Math.random() * titles.length)]}`}
                />
                <Button
                    className="max-w-100 bg-neutral-700 align-middle hover:cursor-pointer hover:bg-neutral-600"
                    onClick={() => {
                        const currentResults = findNearestMovie(currentInput);
                        setResults(currentResults);
                    }}
                >
                    Find similar movies
                </Button>
                {results.length > 0 ? (
                    results.map((result, index) => {
                        return (
                            <div className="p-2">
                                <a href={`https://movielens.org/movies/${result.movieID}`} target="_blank">
                                    {index + 1}: {idToTitle(result.movieID)}
                                </a>
                            </div>
                        );
                    })
                ) : (
                    <></>
                )}
            </div>
        </div>
    );
}
