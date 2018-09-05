import os
import argparse
import tmdbsimple as tmdb
import json
import urllib.request


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="TMDB downloader script")
    parser.add_argument("--download_dir", required=True, type=str, help="Dataset directory")
    parser.add_argument("--api_key_file", required=True, type=str, help="Text file containing TMDB api key")
    parser.add_argument("--number_of_posters", "-n", required=True, type=int,
                        help="Number of random posters to download")
    args = parser.parse_args()
    return args


def main():

    # TODO: Support dataset update

    # TODO: Create a script to split train and val (create 2 annotations files)

    # Load params
    args = do_parsing()
    print(args)

    with open(args.api_key_file, "r") as api_file:
        api_key = api_file.readlines()[0]

    tmdb.API_KEY = api_key

    os.makedirs(os.path.join(args.download_dir, "images"), exist_ok=False)
    os.makedirs(os.path.join(args.download_dir, "annotations"), exist_ok=False)

    samples = []

    # Retrieve the Top N popular movies
    downloaded = 0
    page = 1
    total_results = tmdb.Movies().top_rated()["total_results"]
    print("Total results with top_rated API: " + str(total_results))

    # FIXME: Download exactly args.number_of_posters images
    while downloaded <= args.number_of_posters:
        print("Downloading page: " + str(page))
        movies = tmdb.Movies().top_rated(page=page)
        for movie in movies["results"]:
            movie_id = str(movie["id"])
            title = str(movie["title"])
            print("Title: " + title)
            urllib.request.urlretrieve("http://image.tmdb.org/t/p/original" + movie["poster_path"],
                                       os.path.join(args.download_dir, "images", str(movie["id"]) + ".jpg"))
            samples.append({"movie_id": movie_id, "title":title})
            downloaded += 1
        page += 1

    print("Writing captions file")
    with open(os.path.join(args.download_dir, "annotations", "full_captions.json"), "w") as caption_out:
        caption_out.write(json.dumps(samples))

    print("End")


if __name__ == "__main__":
    main()
