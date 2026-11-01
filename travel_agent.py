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
    state: str = ""


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

    ALL_STATES_AND_UTS = (
        "Andhra Pradesh",
        "Arunachal Pradesh",
        "Assam",
        "Bihar",
        "Chhattisgarh",
        "Goa",
        "Gujarat",
        "Haryana",
        "Himachal Pradesh",
        "Jharkhand",
        "Karnataka",
        "Kerala",
        "Madhya Pradesh",
        "Maharashtra",
        "Manipur",
        "Meghalaya",
        "Mizoram",
        "Nagaland",
        "Odisha",
        "Punjab",
        "Rajasthan",
        "Sikkim",
        "Tamil Nadu",
        "Telangana",
        "Tripura",
        "Uttar Pradesh",
        "Uttarakhand",
        "West Bengal",
        "Andaman and Nicobar Islands",
        "Chandigarh",
        "Dadra and Nagar Haveli and Daman and Diu",
        "Delhi",
        "Jammu and Kashmir",
        "Ladakh",
        "Lakshadweep",
        "Puducherry",
    )

    def __init__(self) -> None:
        self.cities: dict[str, City] = {
            # Northern India
            "Delhi": City(
                "Delhi", "India",
                (
                    Attraction("Red Fort", "history", 2, 350, "Mughal-era fort and UNESCO site"),
                    Attraction("India Gate", "history", 1, 0, "Iconic war memorial"),
                    Attraction("Chandni Chowk", "food", 3, 800, "Historic market and street food"),
                ),
                (Hotel("Delhi Comfort Inn", 2200, 4.1), Hotel("Delhi Budget Stay", 1300, 3.8)),
                state="Delhi",
            ),
            "Chandigarh": City(
                "Chandigarh", "India",
                (
                    Attraction("Rock Garden", "nature", 2.5, 50, "Sculpture garden made of industrial waste"),
                    Attraction("Sukhna Lake", "nature", 2, 0, "Picturesque reservoir at the Shivalik foothills"),
                    Attraction("Sector 17 Plaza", "shopping", 2, 300, "Popular shopping and pedestrian plaza"),
                ),
                (Hotel("Chandigarh Grand", 2500, 4.2), Hotel("City Stay Chandigarh", 1400, 3.9)),
                state="Chandigarh / Punjab / Haryana",
            ),
            "Amritsar": City(
                "Amritsar", "India",
                (
                    Attraction("Golden Temple", "history", 3, 0, "Holiest Gurdwara and spiritual center of Sikhism"),
                    Attraction("Jallianwala Bagh", "history", 1.5, 0, "Historic memorial public garden"),
                    Attraction("Wagah Border", "history", 3, 0, "Famous evening military ceremony"),
                ),
                (Hotel("Amritsar Heritage Inn", 2300, 4.3), Hotel("Amritsar Golden Stay", 1200, 3.8)),
                state="Punjab",
            ),
            "Srinagar": City(
                "Srinagar", "India",
                (
                    Attraction("Dal Lake Shikara", "nature", 3, 800, "Scenic wooden boat ride on Dal Lake"),
                    Attraction("Mughal Gardens", "nature", 2, 100, "Terraced gardens with Persian architecture"),
                    Attraction("Lal Chowk Bazaar", "shopping", 2, 500, "Traditional Kashmiri handicrafts and spices"),
                ),
                (Hotel("Dal View Palace", 3200, 4.4), Hotel("Kashmir Pine Retreat", 1600, 3.9)),
                state="Jammu and Kashmir",
            ),
            "Leh": City(
                "Leh", "India",
                (
                    Attraction("Pangong Lake", "nature", 6, 500, "High altitude endorheic mountain lake"),
                    Attraction("Thiksey Monastery", "history", 2, 100, "Tibetan Buddhist monastery atop a hill"),
                    Attraction("Leh Market", "shopping", 2, 400, "Tibetan souvenirs, woolens, and cafes"),
                ),
                (Hotel("Himalayan Residency", 3500, 4.5), Hotel("Ladakh Budget Haven", 1700, 4.0)),
                state="Ladakh",
            ),
            "Shimla": City(
                "Shimla", "India",
                (
                    Attraction("The Ridge & Mall Road", "shopping", 3, 0, "Cultural hub and shopping street"),
                    Attraction("Jakhoo Temple", "history", 2, 0, "Ancient temple with panoramic hill views"),
                    Attraction("Kufri Nature Park", "nature", 4, 300, "Scenic viewpoint and snow sports area"),
                ),
                (Hotel("Pine View Resort", 2800, 4.2), Hotel("Shimla Cozy Stay", 1500, 3.8)),
                state="Himachal Pradesh",
            ),
            "Dehradun": City(
                "Dehradun", "India",
                (
                    Attraction("Robber's Cave", "nature", 2.5, 50, "Natural cave river gorge formation"),
                    Attraction("Forest Research Institute", "history", 2, 100, "Greco-Roman architecture and forest museum"),
                    Attraction("Paltan Bazaar", "shopping", 2, 300, "Bustling local market near clock tower"),
                ),
                (Hotel("Doon Valley Retreat", 2400, 4.1), Hotel("Dehradun Traveler Inn", 1300, 3.8)),
                state="Uttarakhand",
            ),
            "Agra": City(
                "Agra", "India",
                (
                    Attraction("Taj Mahal", "history", 3, 1100, "World-famous marble mausoleum"),
                    Attraction("Agra Fort", "history", 2, 500, "Historic Mughal fort"),
                    Attraction("Sadar Bazaar", "shopping", 2, 400, "Local crafts and leather goods"),
                ),
                (Hotel("Agra View Hotel", 2100, 4.2), Hotel("Agra Budget Stay", 1200, 3.7)),
                state="Uttar Pradesh",
            ),
            "Varanasi": City(
                "Varanasi", "India",
                (
                    Attraction("Dashashwamedh Ghat Ganga Aarti", "history", 2.5, 0, "Mesmerizing evening spiritual ritual"),
                    Attraction("Kashi Vishwanath Temple", "history", 2, 200, "Ancient revered Shiva temple"),
                    Attraction("Kachori Gali", "food", 2, 300, "Famous local street food and sweets lane"),
                ),
                (Hotel("Ganga Heritage Stay", 2400, 4.3), Hotel("Kashi Budget Inn", 1200, 3.8)),
                state="Uttar Pradesh",
            ),

            # Western & Central India
            "Jaipur": City(
                "Jaipur", "India",
                (
                    Attraction("Amber Fort", "history", 3, 550, "Hilltop fort with palace courtyards"),
                    Attraction("Hawa Mahal", "history", 1.5, 250, "Pink sandstone palace facade"),
                    Attraction("Johari Bazaar", "shopping", 2, 500, "Traditional jewelry and textile market"),
                ),
                (Hotel("Jaipur Heritage Hotel", 2400, 4.4), Hotel("Jaipur Budget Stay", 1400, 3.9)),
                state="Rajasthan",
            ),
            "Ahmedabad": City(
                "Ahmedabad", "India",
                (
                    Attraction("Sabarmati Ashram", "history", 2, 0, "Mahatma Gandhi's historic residence"),
                    Attraction("Adalaj Stepwell", "history", 1.5, 50, "Intricately carved subterranean water reservoir"),
                    Attraction("Manek Chowk Food Night Market", "food", 2.5, 400, "Famous midnight Gujarati street food hub"),
                ),
                (Hotel("Ahmedabad Central Inn", 2300, 4.2), Hotel("Sabarmati Budget Stay", 1300, 3.8)),
                state="Gujarat",
            ),
            "Bhopal": City(
                "Bhopal", "India",
                (
                    Attraction("Upper Lake (Bhojtal)", "nature", 2, 100, "Sprawling lake with water sports and views"),
                    Attraction("Bhimbetka Caves", "history", 3.5, 250, "Paleolithic and Mesolithic rock art site"),
                    Attraction("Chowk Bazaar", "food", 2, 350, "Historic Old City street food and kebabs"),
                ),
                (Hotel("Lakefront Palace Bhopal", 2600, 4.2), Hotel("Bhopal Budget Inn", 1300, 3.8)),
                state="Madhya Pradesh",
            ),
            "Raipur": City(
                "Raipur", "India",
                (
                    Attraction("Nandan Van Zoo & Safari", "nature", 3, 200, "Wildlife safari and zoological park"),
                    Attraction("Swami Vivekananda Sarovar", "nature", 2, 0, "Peaceful lake with towering Vivekananda statue"),
                    Attraction("Gol Bazaar", "shopping", 2, 300, "Oldest market for handlooms and bell-metal art"),
                ),
                (Hotel("Raipur Grand Hotel", 2300, 4.0), Hotel("Chhattisgarh Comfort Stay", 1250, 3.7)),
                state="Chhattisgarh",
            ),
            "Mumbai": City(
                "Mumbai", "India",
                (
                    Attraction("Gateway of India", "history", 1.5, 0, "Harbour landmark"),
                    Attraction("Marine Drive", "nature", 2, 0, "Seafront promenade"),
                    Attraction("Colaba Causeway", "shopping", 2.5, 700, "Popular shopping district"),
                ),
                (Hotel("Mumbai Central Hotel", 3200, 4.2), Hotel("Mumbai Budget Stay", 1900, 3.8)),
                state="Maharashtra",
            ),
            "Goa": City(
                "Goa", "India",
                (
                    Attraction("Baga Beach", "beach", 4, 0, "Popular beach for swimming and nightlife"),
                    Attraction("Basilica of Bom Jesus", "history", 2, 200, "Historic church in Old Goa"),
                    Attraction("Dudhsagar Falls", "nature", 5, 1200, "Scenic waterfall excursion"),
                ),
                (Hotel("Goa Beach Resort", 3500, 4.5), Hotel("Goa Budget Stay", 1800, 4.0)),
                state="Goa",
            ),

            # Eastern & Northeast India
            "Patna": City(
                "Patna", "India",
                (
                    Attraction("Golghar", "history", 1.5, 50, "Historic beehive-shaped granary offering river views"),
                    Attraction("Bihar Museum", "history", 2.5, 100, "World-class museum of ancient Indian art"),
                    Attraction("Maurya Lok Complex", "food", 2, 350, "Central hub for local snacks and shopping"),
                ),
                (Hotel("Patna Heritage Stay", 2200, 4.1), Hotel("Pataliputra Inn", 1200, 3.7)),
                state="Bihar",
            ),
            "Ranchi": City(
                "Ranchi", "India",
                (
                    Attraction("Dassam Falls", "nature", 3, 100, "Spectacular waterfall formed by the Kanchi River"),
                    Attraction("Tagore Hill", "history", 2, 0, "Scenic hilltop associated with Rabindranath Tagore"),
                    Attraction("Main Road Market", "shopping", 2, 300, "Tribal handicrafts and bamboo items"),
                ),
                (Hotel("Ranchi Pine Inn", 2100, 4.0), Hotel("Ranchi Budget Stay", 1150, 3.7)),
                state="Jharkhand",
            ),
            "Kolkata": City(
                "Kolkata", "India",
                (
                    Attraction("Victoria Memorial", "history", 2.5, 100, "Monumental white marble palace and museum"),
                    Attraction("Howrah Bridge", "history", 1.5, 0, "Iconic cantilever bridge over the Hooghly"),
                    Attraction("Park Street & New Market", "food", 3, 600, "Historic dining street and legendary bakery delicacies"),
                ),
                (Hotel("Kolkata Grand Palace", 2800, 4.3), Hotel("City of Joy Stay", 1400, 3.9)),
                state="West Bengal",
            ),
            "Bhubaneswar": City(
                "Bhubaneswar", "India",
                (
                    Attraction("Lingaraj Temple", "history", 2, 0, "11th-century Kalinga architecture masterpiece"),
                    Attraction("Udayagiri and Khandagiri Caves", "history", 2.5, 100, "Rock-cut cave complexes and carvings"),
                    Attraction("Ekamra Haat", "shopping", 2, 400, "Craft village showcasing Odisha handlooms"),
                ),
                (Hotel("Temple City Retreat", 2400, 4.2), Hotel("Bhubaneswar Budget Inn", 1300, 3.8)),
                state="Odisha",
            ),
            "Guwahati": City(
                "Guwahati", "India",
                (
                    Attraction("Kamakhya Temple", "history", 2.5, 0, "Venerated hill shrine dedicated to Shakti"),
                    Attraction("Brahmaputra River Cruise", "nature", 2, 500, "Sunset ferry across the mighty river"),
                    Attraction("Fancy Bazaar", "shopping", 2, 400, "Assam silk and cane souvenir hub"),
                ),
                (Hotel("Brahmaputra River View", 2700, 4.2), Hotel("Assam Gateway Inn", 1350, 3.8)),
                state="Assam",
            ),
            "Gangtok": City(
                "Gangtok", "India",
                (
                    Attraction("Tsomgo Lake", "nature", 5, 800, "Glacial mountain lake at high altitude"),
                    Attraction("Rumtek Monastery", "history", 2, 50, "Historic Kagyu sect Buddhist monastery"),
                    Attraction("MG Marg Promenade", "shopping", 2.5, 450, "Pedestrian street for momos, cafes, and crafts"),
                ),
                (Hotel("Gangtok Mountain Resort", 3000, 4.4), Hotel("Sikkim Cloud Stay", 1500, 3.9)),
                state="Sikkim",
            ),
            "Shillong": City(
                "Shillong", "India",
                (
                    Attraction("Elephant Falls", "nature", 2, 50, "Three-tiered cascade surrounded by lush fern hills"),
                    Attraction("Umiam Lake", "nature", 3, 200, "Picturesque reservoir offering boating and hills"),
                    Attraction("Police Bazar", "shopping", 2.5, 400, "Bustling downtown shopping and cafes"),
                ),
                (Hotel("Pine Wood Shillong", 2900, 4.3), Hotel("Meghalaya Misty Stay", 1450, 3.9)),
                state="Meghalaya",
            ),
            "Itanagar": City(
                "Itanagar", "India",
                (
                    Attraction("Ita Fort", "history", 2, 50, "14th-century brick fort of the Chutiya kingdom"),
                    Attraction("Ganga Lake (Gyakar Sinyi)", "nature", 2.5, 50, "Clear lake enclosed by primeval forest"),
                    Attraction("Jawaharlal Nehru State Museum", "history", 2, 50, "Tribal culture, textiles, and weapons"),
                ),
                (Hotel("Itanagar Valley Stay", 2200, 4.0), Hotel("Arunachal Inn", 1200, 3.7)),
                state="Arunachal Pradesh",
            ),
            "Kohima": City(
                "Kohima", "India",
                (
                    Attraction("Kohima War Cemetery", "history", 2, 0, "WWII memorial overlooking the town"),
                    Attraction("Kisama Heritage Village", "history", 3, 100, "Traditional Naga village architecture and festival site"),
                    Attraction("Kohima Night Market", "food", 2, 350, "Local Naga delicacies and fresh hill produce"),
                ),
                (Hotel("Highland Hills Kohima", 2500, 4.1), Hotel("Naga Homestay Kohima", 1300, 3.8)),
                state="Nagaland",
            ),
            "Imphal": City(
                "Imphal", "India",
                (
                    Attraction("Kangla Fort", "history", 2.5, 50, "Ancient royal palace of the Meitei kingdom"),
                    Attraction("Loktak Lake & Floating Islands", "nature", 5, 600, "World's only floating national park phumdis"),
                    Attraction("Ima Keithel (Mother's Market)", "shopping", 2, 300, "Asia's largest all-women operated market"),
                ),
                (Hotel("Imphal Classic Hotel", 2600, 4.2), Hotel("Manipur Guest House", 1300, 3.8)),
                state="Manipur",
            ),
            "Aizawl": City(
                "Aizawl", "India",
                (
                    Attraction("Durtlang Hills", "nature", 2.5, 0, "Breathtaking panoramic ridge viewpoint"),
                    Attraction("Solomon's Temple", "history", 2, 0, "Magnificent marble church complex"),
                    Attraction("Bara Bazar", "shopping", 2, 350, "Mizo weaves, bamboo baskets, and spices"),
                ),
                (Hotel("Aizawl Hill View", 2400, 4.1), Hotel("Mizoram Budget Stay", 1250, 3.7)),
                state="Mizoram",
            ),
            "Agartala": City(
                "Agartala", "India",
                (
                    Attraction("Ujjayanta Palace", "history", 2.5, 100, "Neoclassical royal palace with Mughal gardens"),
                    Attraction("Neermahal Water Palace", "history", 4, 300, "Palace situated in the middle of Rudrasagar Lake"),
                    Attraction("Battala Market", "shopping", 2, 300, "Local Tripura cane handicrafts and textiles"),
                ),
                (Hotel("Agartala Heritage Stay", 2200, 4.0), Hotel("Tripura City Inn", 1200, 3.7)),
                state="Tripura",
            ),

            # Southern India
            "Hyderabad": City(
                "Hyderabad", "India",
                (
                    Attraction("Charminar", "history", 2, 50, "16th-century landmark with four minarets"),
                    Attraction("Golconda Fort", "history", 3, 200, "Massive fortified citadel and acoustic marvel"),
                    Attraction("Paradise Biryani & Laad Bazaar", "food", 2.5, 600, "World-renowned Hyderabadi biryani and pearls"),
                ),
                (Hotel("Hyderabad City Center Hotel", 2700, 4.3), Hotel("Nizam Budget Inn", 1400, 3.9)),
                state="Telangana",
            ),
            "Visakhapatnam": City(
                "Visakhapatnam", "India",
                (
                    Attraction("R.K. Beach & Submarine Museum", "beach", 3, 100, "Naval museum inside a decommissioned submarine"),
                    Attraction("Kailasagiri Hilltop Park", "nature", 2.5, 150, "Hilltop park with coastal bay vistas"),
                    Attraction("Jagadamba Centre", "shopping", 2, 400, "Main retail shopping and coastal Andhra food"),
                ),
                (Hotel("Vizag Ocean View", 2800, 4.2), Hotel("Andhra Bay Stay", 1350, 3.8)),
                state="Andhra Pradesh",
            ),
            "Bengaluru": City(
                "Bengaluru", "India",
                (
                    Attraction("Lalbagh Botanical Garden", "nature", 3, 100, "Large botanical garden"),
                    Attraction("Bangalore Palace", "history", 2, 500, "Tudor-style palace"),
                    Attraction("VV Puram Food Street", "food", 2, 700, "Street-food destination"),
                ),
                (Hotel("Bengaluru City Hotel", 2300, 4.1), Hotel("Bengaluru Budget Stay", 1350, 3.8)),
                state="Karnataka",
            ),
            "Chennai": City(
                "Chennai", "India",
                (
                    Attraction("Marina Beach", "beach", 3, 0, "One of the longest urban natural beaches"),
                    Attraction("Kapaleeshwarar Temple", "history", 2, 0, "Dravidian architecture temple in Mylapore"),
                    Attraction("T. Nagar & Murugan Idli", "food", 2.5, 450, "Silk sarees and authentic South Indian breakfast"),
                ),
                (Hotel("Chennai Coastline Grand", 2800, 4.3), Hotel("Chennai Budget Inn", 1400, 3.9)),
                state="Tamil Nadu",
            ),
            "Kochi": City(
                "Kochi", "India",
                (
                    Attraction("Fort Kochi Chinese Fishing Nets", "beach", 2.5, 0, "Colonial history, art, and cantilever nets"),
                    Attraction("Mattancherry Palace", "history", 1.5, 50, "Portuguese palace with Kerala murals"),
                    Attraction("Jew Town & Spice Market", "shopping", 2, 500, "Antiques, aromatic spices, and cafes"),
                ),
                (Hotel("Kochi Backwaters Resort", 3100, 4.4), Hotel("Cochin Budget Stay", 1500, 3.9)),
                state="Kerala",
            ),
            "Puducherry": City(
                "Puducherry", "India",
                (
                    Attraction("Promenade Beach", "beach", 2.5, 0, "French quarter seafront promenade"),
                    Attraction("Auroville Matrimandir", "nature", 3, 0, "Experimental township and meditation dome"),
                    Attraction("French Quarter Bakeries", "food", 2, 450, "Croissants, baguettes, and cafe culture"),
                ),
                (Hotel("French Colony Villa", 3000, 4.4), Hotel("Pondy Seaside Stay", 1500, 3.9)),
                state="Puducherry",
            ),
            "Port Blair": City(
                "Port Blair", "India",
                (
                    Attraction("Cellular Jail National Memorial", "history", 2.5, 100, "Historic colonial prison with light and sound show"),
                    Attraction("Corbyn's Cove Beach", "beach", 3, 0, "Coconut-palm fringed sandy beach"),
                    Attraction("Aberdeen Bazaar", "shopping", 2, 400, "Shell handicrafts and coastal seafood"),
                ),
                (Hotel("Andaman Palms Resort", 3500, 4.4), Hotel("Island Budget Stay", 1800, 3.9)),
                state="Andaman and Nicobar Islands",
            ),
        }

        # Undirected connections are stored in both directions for graph search.
        self.transport: dict[str, list[TransportOption]] = {city: [] for city in self.cities}

        # Northern hub connections
        self._add_connection("Delhi", "Chandigarh", "train", 3.5, 600)
        self._add_connection("Delhi", "Agra", "train", 3, 700)
        self._add_connection("Delhi", "Jaipur", "train", 5, 900)
        self._add_connection("Delhi", "Shimla", "bus", 7.5, 900)
        self._add_connection("Delhi", "Dehradun", "train", 5.5, 800)
        self._add_connection("Delhi", "Varanasi", "train", 8.5, 1200)
        self._add_connection("Delhi", "Srinagar", "flight", 1.5, 3800)
        self._add_connection("Delhi", "Leh", "flight", 1.5, 4500)
        self._add_connection("Delhi", "Ahmedabad", "train", 13, 1400)
        self._add_connection("Delhi", "Bhopal", "train", 8, 1100)
        self._add_connection("Delhi", "Patna", "train", 12, 1300)
        self._add_connection("Delhi", "Kolkata", "flight", 2.2, 4200)
        self._add_connection("Delhi", "Mumbai", "flight", 2.2, 4500)
        self._add_connection("Delhi", "Hyderabad", "flight", 2.3, 4200)
        self._add_connection("Delhi", "Bengaluru", "flight", 2.7, 5000)
        self._add_connection("Delhi", "Chennai", "flight", 2.8, 4800)
        self._add_connection("Delhi", "Guwahati", "flight", 2.5, 4600)

        self._add_connection("Chandigarh", "Amritsar", "train", 3.5, 450)
        self._add_connection("Chandigarh", "Shimla", "bus", 4, 500)
        self._add_connection("Amritsar", "Srinagar", "flight", 1.2, 3200)
        self._add_connection("Srinagar", "Leh", "bus", 10, 1800)
        self._add_connection("Shimla", "Dehradun", "bus", 6, 700)
        self._add_connection("Dehradun", "Agra", "train", 7, 750)

        # Central & Western routes
        self._add_connection("Jaipur", "Agra", "bus", 5, 1000)
        self._add_connection("Jaipur", "Ahmedabad", "train", 9, 950)
        self._add_connection("Jaipur", "Mumbai", "flight", 2, 4000)
        self._add_connection("Agra", "Varanasi", "train", 9, 900)
        self._add_connection("Agra", "Bhopal", "train", 6.5, 800)
        self._add_connection("Agra", "Mumbai", "train", 18, 2200)
        self._add_connection("Ahmedabad", "Mumbai", "train", 6, 900)
        self._add_connection("Ahmedabad", "Bhopal", "train", 8.5, 850)
        self._add_connection("Bhopal", "Raipur", "train", 9, 850)
        self._add_connection("Bhopal", "Mumbai", "train", 12, 1100)
        self._add_connection("Bhopal", "Hyderabad", "train", 12, 1200)
        self._add_connection("Mumbai", "Goa", "flight", 1.2, 2500)
        self._add_connection("Mumbai", "Bengaluru", "flight", 1.6, 3000)
        self._add_connection("Mumbai", "Hyderabad", "train", 13, 1100)
        self._add_connection("Goa", "Bengaluru", "bus", 11, 1300)
        self._add_connection("Goa", "Kochi", "train", 14, 1200)

        # Eastern & Central routes
        self._add_connection("Varanasi", "Patna", "train", 4, 450)
        self._add_connection("Patna", "Ranchi", "train", 6.5, 550)
        self._add_connection("Patna", "Kolkata", "train", 8, 800)
        self._add_connection("Ranchi", "Raipur", "train", 9, 750)
        self._add_connection("Ranchi", "Kolkata", "train", 7, 600)
        self._add_connection("Ranchi", "Bhubaneswar", "train", 7.5, 650)
        self._add_connection("Raipur", "Bhubaneswar", "train", 10, 800)
        self._add_connection("Raipur", "Visakhapatnam", "train", 9.5, 750)
        self._add_connection("Kolkata", "Bhubaneswar", "train", 6.5, 700)
        self._add_connection("Kolkata", "Gangtok", "flight", 1.2, 3200)
        self._add_connection("Kolkata", "Guwahati", "flight", 1.2, 3200)
        self._add_connection("Kolkata", "Agartala", "flight", 1.0, 2600)
        self._add_connection("Kolkata", "Port Blair", "flight", 2.0, 4800)
        self._add_connection("Kolkata", "Chennai", "flight", 2.2, 4200)
        self._add_connection("Bhubaneswar", "Visakhapatnam", "train", 6.5, 650)

        # Northeast Network
        self._add_connection("Guwahati", "Shillong", "bus", 3, 350)
        self._add_connection("Guwahati", "Gangtok", "train", 10, 950)
        self._add_connection("Guwahati", "Itanagar", "train", 7, 600)
        self._add_connection("Guwahati", "Kohima", "train", 7.5, 650)
        self._add_connection("Guwahati", "Imphal", "flight", 1.0, 2800)
        self._add_connection("Guwahati", "Agartala", "flight", 1.0, 2600)
        self._add_connection("Kohima", "Imphal", "bus", 4.5, 450)
        self._add_connection("Imphal", "Aizawl", "bus", 9, 900)
        self._add_connection("Aizawl", "Agartala", "bus", 8, 800)

        # Southern Network
        self._add_connection("Visakhapatnam", "Hyderabad", "train", 11, 950)
        self._add_connection("Visakhapatnam", "Chennai", "train", 12, 1100)
        self._add_connection("Hyderabad", "Bengaluru", "train", 9.5, 950)
        self._add_connection("Hyderabad", "Chennai", "train", 12, 1000)
        self._add_connection("Bengaluru", "Chennai", "train", 5, 600)
        self._add_connection("Bengaluru", "Kochi", "flight", 1.1, 2400)
        self._add_connection("Chennai", "Puducherry", "bus", 3.5, 300)
        self._add_connection("Chennai", "Kochi", "train", 11, 1000)
        self._add_connection("Chennai", "Port Blair", "flight", 2.1, 4900)
        self._add_connection("Puducherry", "Kochi", "bus", 10, 950)

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

