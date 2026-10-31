"""A small, explainable intelligent travel-planning agent.

The project deliberately keeps the knowledge base in Python data structures so it
is easy to extend for a college project.  It demonstrates:

* a knowledge base of cities, attractions, hotels, transport, and relationships;
* eligibility rules based on interests, time, and budget;
* BFS, DFS, Uniform Cost, Greedy Best-First, and A* route search;
* a goal-based agent and a utility-based agent;
* an itinerary with totals and human-readable reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from math import inf
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class Attraction:
    name: str
    category: str
    visit_hours: float
    cost: float
    description: str


@dataclass(frozen=True)
class Hotel:
    name: str
    nightly_cost: float
    rating: float


@dataclass(frozen=True)
class City:
    name: str
    country: str
    attractions: tuple[Attraction, ...]
    hotels: tuple[Hotel, ...]


@dataclass(frozen=True)
class TransportOption:
    origin: str
    destination: str
    mode: str
    hours: float
    cost: float


@dataclass(frozen=True)
class Route:
    cities: tuple[str, ...]
    legs: tuple[TransportOption, ...]
    cost: float
    hours: float

    @property
    def stops(self) -> int:
        return max(0, len(self.cities) - 2)


@dataclass(frozen=True)
class TravelRequest:
    origin: str
    destinations: tuple[str, ...]
    budget: float
    available_hours: float
    interests: tuple[str, ...] = ()
    travelers: int = 1
    algorithm: str = "astar"

    @property
    def available_days(self) -> float:
        return self.available_hours / 24


@dataclass
class Itinerary:
    request: TravelRequest
    route: Route
    eligible_destinations: tuple[str, ...]
    rejected_destinations: dict[str, tuple[str, ...]]
    hotels: dict[str, Hotel]
    activities: dict[str, tuple[Attraction, ...]]
    transport_cost: float
    hotel_cost: float
    activity_cost: float
    total_cost: float
    total_hours: float
    reasoning: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        budget_status = "WITHIN BUDGET" if self.total_cost <= self.request.budget else "OVER BUDGET"
        lines = [
            "TRAVEL AGENT ITINERARY",
            "=" * 24,
            f"Route: {' -> '.join(self.route.cities)}",
            f"Search method: {self.request.algorithm.upper()}",
            f"Travel time: {self.route.hours:.1f} hours",
            f"Estimated trip time: {self.total_hours:.1f} hours",
            f"Estimated cost: {self.request_currency(self.total_cost)}",
            f"Budget remaining: {self.request_currency(self.request.budget - self.total_cost)}",
            f"Budget status: {budget_status}",
            "",
            "TRANSPORT DETAILS:",
        ]
        for leg in self.route.legs:
            lines.append(
                f"  {leg.origin} -> {leg.destination} | {leg.mode.title()} | "
                f"{leg.hours:g} h | {self.request_currency(leg.cost * self.request.travelers)}"
            )
        lines.extend([
            "",
            "COST BREAKDOWN:",
            f"  Transport: {self.request_currency(self.transport_cost)}",
            f"  Hotels: {self.request_currency(self.hotel_cost)}",
            f"  Activities: {self.request_currency(self.activity_cost)}",
            f"  Total: {self.request_currency(self.total_cost)}",
            "",
            "PLAN:",
        ])
        for city in self.route.cities[1:]:
            if city not in self.activities:
                continue
            hotel = self.hotels[city]
            lines.append(f"\n{city} - stay at {hotel.name} ({self.request_currency(hotel.nightly_cost)}/night)")
            for attraction in self.activities[city]:
                lines.append(
                    f"  * {attraction.name} [{attraction.category}] - "
                    f"{attraction.visit_hours:g} h, {self.request_currency(attraction.cost)}"
                )
        lines.extend(["", "REASONING:"])
        lines.extend(f"  * {item}" for item in self.reasoning)
        return "\n".join(lines)

    @staticmethod
    def request_currency(value: float) -> str:
        # Keep CLI output safe on Windows consoles that use cp1252 by default.
        return f"INR {value:,.0f}"


class TravelKnowledgeBase:
    """Facts and relationships used by the agents."""

    def __init__(self) -> None:
        self.cities: dict[str, City] = {
            "Delhi": City(
                "Delhi", "India",
                (
                    Attraction("Red Fort", "history", 2, 350, "Mughal-era fort and UNESCO site"),
                    Attraction("India Gate", "history", 1, 0, "Iconic war memorial"),
                    Attraction("Chandni Chowk", "food", 3, 800, "Historic market and street food"),
                ),
                (Hotel("Delhi Comfort Inn", 2200, 4.1), Hotel("Delhi Budget Stay", 1300, 3.8)),
            ),
            "Jaipur": City(
                "Jaipur", "India",
                (
                    Attraction("Amber Fort", "history", 3, 550, "Hilltop fort with palace courtyards"),
                    Attraction("Hawa Mahal", "history", 1.5, 250, "Pink sandstone palace facade"),
                    Attraction("Johari Bazaar", "shopping", 2, 500, "Traditional jewelry and textile market"),
                ),
                (Hotel("Jaipur Heritage Hotel", 2400, 4.4), Hotel("Jaipur Budget Stay", 1400, 3.9)),
            ),
            "Agra": City(
                "Agra", "India",
                (
                    Attraction("Taj Mahal", "history", 3, 1100, "World-famous marble mausoleum"),
                    Attraction("Agra Fort", "history", 2, 500, "Historic Mughal fort"),
                    Attraction("Sadar Bazaar", "shopping", 2, 400, "Local crafts and leather goods"),
                ),
                (Hotel("Agra View Hotel", 2100, 4.2), Hotel("Agra Budget Stay", 1200, 3.7)),
            ),
            "Mumbai": City(
                "Mumbai", "India",
                (
                    Attraction("Gateway of India", "history", 1.5, 0, "Harbour landmark"),
                    Attraction("Marine Drive", "nature", 2, 0, "Seafront promenade"),
                    Attraction("Colaba Causeway", "shopping", 2.5, 700, "Popular shopping district"),
                ),
                (Hotel("Mumbai Central Hotel", 3200, 4.2), Hotel("Mumbai Budget Stay", 1900, 3.8)),
            ),
            "Goa": City(
                "Goa", "India",
                (
                    Attraction("Baga Beach", "beach", 4, 0, "Popular beach for swimming and nightlife"),
                    Attraction("Basilica of Bom Jesus", "history", 2, 200, "Historic church in Old Goa"),
                    Attraction("Dudhsagar Falls", "nature", 5, 1200, "Scenic waterfall excursion"),
                ),
                (Hotel("Goa Beach Resort", 3500, 4.5), Hotel("Goa Budget Stay", 1800, 4.0)),
            ),
            "Bengaluru": City(
                "Bengaluru", "India",
                (
                    Attraction("Lalbagh Botanical Garden", "nature", 3, 100, "Large botanical garden"),
                    Attraction("Bangalore Palace", "history", 2, 500, "Tudor-style palace"),
                    Attraction("VV Puram Food Street", "food", 2, 700, "Street-food destination"),
                ),
                (Hotel("Bengaluru City Hotel", 2300, 4.1), Hotel("Bengaluru Budget Stay", 1350, 3.8)),
            ),
        }

        # Undirected connections are stored in both directions for graph search.
        self.transport: dict[str, list[TransportOption]] = {city: [] for city in self.cities}
        self._add_connection("Delhi", "Jaipur", "train", 5, 900)
        self._add_connection("Delhi", "Agra", "train", 3, 700)
        self._add_connection("Delhi", "Mumbai", "flight", 2.2, 4500)
        self._add_connection("Delhi", "Bengaluru", "flight", 2.7, 5000)
        self._add_connection("Jaipur", "Agra", "bus", 5, 1000)
        self._add_connection("Jaipur", "Mumbai", "flight", 2, 4000)
        self._add_connection("Agra", "Mumbai", "train", 18, 2200)
        self._add_connection("Mumbai", "Goa", "flight", 1.2, 2500)
        self._add_connection("Mumbai", "Bengaluru", "flight", 1.6, 3000)
        self._add_connection("Goa", "Bengaluru", "bus", 11, 1300)

    def _add_connection(self, first: str, second: str, mode: str, hours: float, cost: float) -> None:
        self.transport[first].append(TransportOption(first, second, mode, hours, cost))
        self.transport[second].append(TransportOption(second, first, mode, hours, cost))

    def neighbors(self, city: str) -> tuple[TransportOption, ...]:
        return tuple(sorted(self.transport.get(city, ()), key=lambda option: option.destination))

    def get_city(self, city: str) -> City:
        try:
            return self.cities[city]
        except KeyError as exc:
            raise ValueError(f"Unknown city: {city}. Choose from {', '.join(sorted(self.cities))}.") from exc


def _validate_endpoints(kb: TravelKnowledgeBase, start: str, goal: str) -> None:
    kb.get_city(start)
    kb.get_city(goal)


def _route_from_path(kb: TravelKnowledgeBase, path: Sequence[str]) -> Route:
    legs: list[TransportOption] = []
    for origin, destination in zip(path, path[1:]):
        leg = next((item for item in kb.neighbors(origin) if item.destination == destination), None)
        if leg is None:
            raise ValueError(f"No connection from {origin} to {destination}.")
        legs.append(leg)
    return Route(tuple(path), tuple(legs), sum(leg.cost for leg in legs), sum(leg.hours for leg in legs))


def _reconstruct(parent: dict[str, str | None], goal: str) -> tuple[str, ...]:
    path: list[str] = []
    current: str | None = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    return tuple(reversed(path))


def bfs_route(kb: TravelKnowledgeBase, start: str, goal: str) -> Route:
    """Find a route with the fewest graph edges."""
    _validate_endpoints(kb, start, goal)
    frontier = [start]
    parent: dict[str, str | None] = {start: None}
    while frontier:
        current = frontier.pop(0)
        if current == goal:
            return _route_from_path(kb, _reconstruct(parent, goal))
        for leg in kb.neighbors(current):
            if leg.destination not in parent:
                parent[leg.destination] = current
                frontier.append(leg.destination)
    raise ValueError(f"No route from {start} to {goal}.")


def dfs_route(kb: TravelKnowledgeBase, start: str, goal: str) -> Route:
    """Find any route by depth-first graph exploration."""
    _validate_endpoints(kb, start, goal)
    frontier = [start]
    parent: dict[str, str | None] = {start: None}
    while frontier:
        current = frontier.pop()
        if current == goal:
            return _route_from_path(kb, _reconstruct(parent, goal))
        for leg in reversed(kb.neighbors(current)):
            if leg.destination not in parent:
                parent[leg.destination] = current
                frontier.append(leg.destination)
    raise ValueError(f"No route from {start} to {goal}.")


def uniform_cost_route(kb: TravelKnowledgeBase, start: str, goal: str) -> Route:
    """Find the least-cost route using Dijkstra/UCS."""
    _validate_endpoints(kb, start, goal)
    frontier: list[tuple[float, str]] = [(0, start)]
    parent: dict[str, str | None] = {start: None}
    best_cost = {start: 0.0}
    while frontier:
        frontier.sort(key=lambda item: (item[0], item[1]))
        current_cost, current = frontier.pop(0)
        if current == goal:
            return _route_from_path(kb, _reconstruct(parent, goal))
        if current_cost > best_cost[current]:
            continue
        for leg in kb.neighbors(current):
            new_cost = current_cost + leg.cost
            if new_cost < best_cost.get(leg.destination, inf):
                best_cost[leg.destination] = new_cost
                parent[leg.destination] = current
                frontier.append((new_cost, leg.destination))
    raise ValueError(f"No route from {start} to {goal}.")


def _default_heuristic(kb: TravelKnowledgeBase, city: str, goal: str) -> float:
    """A safe lower bound: any non-goal path needs at least one graph edge."""
    if city == goal:
        return 0.0
    # Using the global cheapest edge avoids overestimating a route that exits the
    # current city through a different, cheaper connection.
    return min((leg.cost for options in kb.transport.values() for leg in options), default=0.0)


def greedy_route(kb: TravelKnowledgeBase, start: str, goal: str) -> Route:
    """Find a route by choosing the node with the smallest heuristic."""
    _validate_endpoints(kb, start, goal)
    frontier = [(0.0, start)]
    parent: dict[str, str | None] = {start: None}
    visited: set[str] = set()
    while frontier:
        frontier.sort(key=lambda item: (item[0], item[1]))
        _, current = frontier.pop(0)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return _route_from_path(kb, _reconstruct(parent, goal))
        for leg in kb.neighbors(current):
            if leg.destination not in visited and leg.destination not in parent:
                parent[leg.destination] = current
                frontier.append((_default_heuristic(kb, leg.destination, goal), leg.destination))
    raise ValueError(f"No route from {start} to {goal}.")


def astar_route(kb: TravelKnowledgeBase, start: str, goal: str) -> Route:
    """Find a low-cost route using f(n) = g(n) + h(n)."""
    _validate_endpoints(kb, start, goal)
    frontier = [(0.0, 0.0, start)]
    parent: dict[str, str | None] = {start: None}
    best_cost = {start: 0.0}
    while frontier:
        frontier.sort(key=lambda item: (item[0], item[2]))
        _, current_cost, current = frontier.pop(0)
        if current == goal:
            return _route_from_path(kb, _reconstruct(parent, goal))
        if current_cost > best_cost[current]:
            continue
        for leg in kb.neighbors(current):
            new_cost = current_cost + leg.cost
            if new_cost < best_cost.get(leg.destination, inf):
                best_cost[leg.destination] = new_cost
                parent[leg.destination] = current
                estimate = new_cost + _default_heuristic(kb, leg.destination, goal)
                frontier.append((estimate, new_cost, leg.destination))
    raise ValueError(f"No route from {start} to {goal}.")


SEARCH_ALGORITHMS: dict[str, Callable[[TravelKnowledgeBase, str, str], Route]] = {
    "bfs": bfs_route,
    "dfs": dfs_route,
    "ucs": uniform_cost_route,
    "uniform-cost": uniform_cost_route,
    "greedy": greedy_route,
    "astar": astar_route,
    "a*": astar_route,
}


def choose_route(kb: TravelKnowledgeBase, start: str, goal: str, algorithm: str) -> Route:
    try:
        search = SEARCH_ALGORITHMS[algorithm.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Use: {', '.join(sorted(SEARCH_ALGORITHMS))}.") from exc
    return search(kb, start, goal)


class BaseTravelAgent:
    def __init__(self, knowledge_base: TravelKnowledgeBase | None = None) -> None:
        self.kb = knowledge_base or TravelKnowledgeBase()

    def evaluate_destination(self, city_name: str, request: TravelRequest) -> tuple[bool, tuple[str, ...], tuple[Attraction, ...]]:
        city = self.kb.get_city(city_name)
        interests = {interest.lower().strip() for interest in request.interests}
        activities = tuple(
            attraction for attraction in city.attractions
            if not interests or attraction.category.lower() in interests
        )
        reasons: list[str] = []
        if not activities:
            reasons.append("no attraction matches the requested interests")
        if activities and sum(item.visit_hours for item in activities) > request.available_hours:
            reasons.append("matching attractions need more time than is available")
        cheapest_hotel = min(city.hotels, key=lambda hotel: hotel.nightly_cost)
        if cheapest_hotel.nightly_cost * max(1, int(request.available_days)) > request.budget:
            reasons.append("the cheapest hotel exceeds the available budget for the stay")
        return not reasons, tuple(reasons), activities

    def eligible_destinations(self, request: TravelRequest) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]], dict[str, tuple[Attraction, ...]]]:
        eligible: list[str] = []
        rejected: dict[str, tuple[str, ...]] = {}
        activities: dict[str, tuple[Attraction, ...]] = {}
        for city in request.destinations:
            valid, reasons, matching = self.evaluate_destination(city, request)
            if valid:
                eligible.append(city)
                activities[city] = matching
            else:
                rejected[city] = reasons
        return tuple(eligible), rejected, activities

    def select_activities(self, city_name: str, request: TravelRequest) -> tuple[Attraction, ...]:
        city = self.kb.get_city(city_name)
        interests = {interest.lower().strip() for interest in request.interests}
        options = [item for item in city.attractions if not interests or item.category.lower() in interests]
        return tuple(sorted(options, key=lambda item: (item.cost, -item.visit_hours))[:3])

    def select_hotel(self, city_name: str, request: TravelRequest) -> Hotel:
        city = self.kb.get_city(city_name)
        return min(city.hotels, key=lambda hotel: (hotel.nightly_cost, -hotel.rating))


class GoalBasedAgent(BaseTravelAgent):
    """First satisfies hard goals, then chooses the cheapest feasible plan."""

    def plan(self, request: TravelRequest) -> Itinerary:
        eligible, rejected, matching = self.eligible_destinations(request)
        if not eligible:
            raise ValueError("No requested destination is eligible. " + self._rejection_text(rejected))

        # Select a sequence that visits every eligible requested destination.
        sequence = self._best_sequence(request.origin, eligible, request.algorithm)
        route = self._route_for_sequence(sequence, request.algorithm)
        hotels = {city: self.select_hotel(city, request) for city in eligible}
        activities = {city: self.select_activities(city, request) for city in eligible}
        hotel_nights = max(1, int(request.available_days) // max(1, len(eligible)))
        hotel_cost = sum(hotel.nightly_cost * hotel_nights * request.travelers for hotel in hotels.values())
        activity_cost = sum(item.cost * request.travelers for items in activities.values() for item in items)
        transport_cost = route.cost * request.travelers
        total_cost = transport_cost + hotel_cost + activity_cost
        total_hours = route.hours + sum(item.visit_hours for items in activities.values() for item in items)
        reasoning = [
            f"Goal-based reasoning kept {len(eligible)} destination(s) that satisfy the rules.",
            f"The {request.algorithm.upper()} search selected a route with {route.stops} intermediate stop(s).",
            "The least expensive available hotel was chosen in each eligible city.",
        ]
        if rejected:
            reasoning.append("Rejected destinations: " + self._rejection_text(rejected))
        if total_cost > request.budget:
            reasoning.append("Warning: all hard destination goals fit, but the complete estimate exceeds the budget.")
        return Itinerary(request, route, eligible, rejected, hotels, activities, transport_cost, hotel_cost, activity_cost, total_cost, total_hours, reasoning)

    def _best_sequence(self, origin: str, destinations: Sequence[str], algorithm: str) -> tuple[str, ...]:
        if len(destinations) == 1:
            return (origin, destinations[0])
        best: tuple[float, tuple[str, ...]] | None = None
        for order in permutations(destinations):
            route = self._route_for_sequence((origin, *order), algorithm)
            candidate = (route.cost, route.cities)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        return best[1]

    def _route_for_sequence(self, sequence: Sequence[str], algorithm: str) -> Route:
        all_cities: list[str] = [sequence[0]]
        all_legs: list[TransportOption] = []
        for start, goal in zip(sequence, sequence[1:]):
            segment = choose_route(self.kb, start, goal, algorithm)
            all_cities.extend(segment.cities[1:])
            all_legs.extend(segment.legs)
        return Route(tuple(all_cities), tuple(all_legs), sum(leg.cost for leg in all_legs), sum(leg.hours for leg in all_legs))

    @staticmethod
    def _rejection_text(rejected: dict[str, tuple[str, ...]]) -> str:
        return "; ".join(f"{city}: {', '.join(reasons)}" for city, reasons in rejected.items())


class UtilityBasedAgent(GoalBasedAgent):
    """Scores feasible plans to balance cost, time, and preference satisfaction."""

    def plan(self, request: TravelRequest) -> Itinerary:
        eligible, rejected, _ = self.eligible_destinations(request)
        if not eligible:
            raise ValueError("No requested destination is eligible. " + self._rejection_text(rejected))
        candidates: list[tuple[float, Itinerary]] = []
        destination_orders = permutations(eligible) if len(eligible) > 1 else [eligible]
        for order in destination_orders:
            sequence = (request.origin, *order)
            route = self._route_for_sequence(sequence, request.algorithm)
            hotels = {city: self.select_hotel(city, request) for city in eligible}
            activities = {city: self.select_activities(city, request) for city in eligible}
            nights = max(1, int(request.available_days) // max(1, len(eligible)))
            hotel_cost = sum(hotel.nightly_cost * nights * request.travelers for hotel in hotels.values())
            activity_cost = sum(item.cost * request.travelers for items in activities.values() for item in items)
            transport_cost = route.cost * request.travelers
            total_cost = transport_cost + hotel_cost + activity_cost
            total_hours = route.hours + sum(item.visit_hours for items in activities.values() for item in items)
            score = self.utility_score(request, total_cost, total_hours, activities, route)
            candidates.append((score, Itinerary(request, route, eligible, rejected, hotels, activities, transport_cost, hotel_cost, activity_cost, total_cost, total_hours, [])))

        score, itinerary = max(candidates, key=lambda item: (item[0], -item[1].total_cost, item[1].route.cities))
        budget_status = "within" if itinerary.total_cost <= request.budget else "over"
        itinerary.reasoning = [
            f"Utility-based reasoning scored {len(candidates)} possible destination order(s).",
            f"The selected plan balances preference matches, cost, and time (utility {score:.3f}).",
            f"The estimate is {budget_status} the requested budget.",
        ]
        if rejected:
            itinerary.reasoning.append("Rejected destinations: " + self._rejection_text(rejected))
        return itinerary

    @staticmethod
    def utility_score(request: TravelRequest, total_cost: float, total_hours: float, activities: dict[str, tuple[Attraction, ...]], route: Route) -> float:
        budget_score = max(0.0, 1.0 - max(0.0, total_cost - request.budget) / max(request.budget, 1))
        time_score = max(0.0, 1.0 - max(0.0, total_hours - request.available_hours) / max(request.available_hours, 1))
        preference_score = min(1.0, sum(len(items) for items in activities.values()) / max(1, len(activities) * 3))
        route_score = 1.0 / (1.0 + route.stops)
        return 0.45 * budget_score + 0.25 * time_score + 0.20 * preference_score + 0.10 * route_score


def build_agent(kind: str, knowledge_base: TravelKnowledgeBase | None = None) -> BaseTravelAgent:
    kind = kind.lower()
    if kind in {"goal", "goal-based", "goal_based"}:
        return GoalBasedAgent(knowledge_base)
    if kind in {"utility", "utility-based", "utility_based"}:
        return UtilityBasedAgent(knowledge_base)
    raise ValueError("Unknown agent type. Choose 'goal' or 'utility'.")


def make_request(
    origin: str,
    destinations: Iterable[str],
    budget: float,
    days: float,
    interests: Iterable[str] = (),
    travelers: int = 1,
    algorithm: str = "astar",
) -> TravelRequest:
    if budget <= 0 or days <= 0 or travelers <= 0:
        raise ValueError("Budget, days, and travelers must be positive.")
    return TravelRequest(origin, tuple(destinations), float(budget), float(days) * 24, tuple(interests), travelers, algorithm)

