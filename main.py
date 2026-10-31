"""Command-line interface for the intelligent travel-planning agent."""

from __future__ import annotations

import argparse
import sys

from travel_agent import TravelKnowledgeBase, build_agent, make_request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an explainable travel plan using classic AI search algorithms."
    )
    parser.add_argument("--from", dest="origin", default="Delhi", help="Starting city")
    parser.add_argument("--to", dest="destinations", nargs="+", default=["Jaipur", "Agra"], help="One or more destinations")
    parser.add_argument("--budget", type=float, default=25000, help="Total budget in INR")
    parser.add_argument("--days", type=float, default=6, help="Available trip duration")
    parser.add_argument("--interests", default="history,food", help="Comma-separated categories: history, beach, nature, shopping, food")
    parser.add_argument("--travelers", type=int, default=1, help="Number of travelers")
    parser.add_argument("--algorithm", choices=["bfs", "dfs", "ucs", "greedy", "astar"], default="astar")
    parser.add_argument("--agent", choices=["goal", "utility"], default="utility")
    parser.add_argument("--show-kb", action="store_true", help="Print facts stored in the knowledge base and exit")
    parser.add_argument("--cli", action="store_true", help="Use the command-line interface instead of the desktop GUI")
    return parser.parse_args()


def show_knowledge_base(kb: TravelKnowledgeBase) -> None:
    print("CITIES AND KNOWLEDGE BASE FACTS")
    print("=" * 34)
    for city in sorted(kb.cities.values(), key=lambda item: item.name):
        print(f"\n{city.name} ({city.country})")
        print("  Attractions:", ", ".join(f"{item.name} [{item.category}]" for item in city.attractions))
        print("  Hotels:", ", ".join(f"{item.name} INR {item.nightly_cost:g}/night" for item in city.hotels))
        print("  Connections:", ", ".join(f"{item.destination} by {item.mode}" for item in kb.neighbors(city.name)))


def main() -> int:
    args = parse_args()
    kb = TravelKnowledgeBase()
    if args.show_kb:
        show_knowledge_base(kb)
        return 0
    if not args.cli:
        try:
            from gui import launch_gui
        except ImportError as error:
            print(f"GUI could not be loaded: {error}", file=sys.stderr)
            print("Run the command-line version with: python main.py --cli", file=sys.stderr)
            return 1
        return launch_gui(kb)
    try:
        interests = tuple(item.strip() for item in args.interests.split(",") if item.strip())
        request = make_request(args.origin, args.destinations, args.budget, args.days, interests, args.travelers, args.algorithm)
        itinerary = build_agent(args.agent, kb).plan(request)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(itinerary.as_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

