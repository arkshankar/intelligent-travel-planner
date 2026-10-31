# Intelligent Travel Planning Agent

## 1. Problem statement

Travellers need to choose destinations, transport, hotels, activities, and a visit sequence while considering time, budget, and personal interests. This project models that problem as an intelligent-agent task over a graph of connected cities.

## 2. Objectives covered

1. Store facts about cities, attractions, transport, hotels, time, and approximate costs.
2. Represent relationships between cities as a graph.
3. Apply eligibility rules using interests, duration, and hotel affordability.
4. Explore routes with BFS and DFS.
5. Select routes using Uniform Cost Search, Greedy Best-First Search, and A*.
6. Implement a goal-based agent.
7. Implement a utility-based agent.
8. Generate an explainable final itinerary.

## 3. System architecture

```text
User preferences
      |
      v
TravelRequest
      |
      v
Knowledge Base ---> Eligibility Rules
      |                    |
      +------> Graph Search+
                         |
                         v
             Goal / Utility Agent
                         |
                         v
                    Itinerary
```

## 4. Knowledge representation

The knowledge base is implemented with Python dataclasses and dictionaries:

- `City` stores a city name, country, attractions, and hotels.
- `Attraction` stores category, visit duration, entry cost, and description.
- `Hotel` stores nightly price and rating.
- `TransportOption` stores origin, destination, mode, travel time, and cost.
- `transport` is an adjacency-list graph used by the search algorithms.

## 5. Search algorithms

### Breadth-First Search

BFS explores the graph level by level. It generally finds the route with the fewest edges, but it does not optimize transport cost.

### Depth-First Search

DFS explores one branch deeply before backtracking. It demonstrates graph exploration but is not guaranteed to find the cheapest or shortest route.

### Uniform Cost Search

UCS expands the partial route with the lowest accumulated cost. It is appropriate when transport prices are the main objective.

### Greedy Best-First Search

Greedy search expands the city with the most promising heuristic value. It can be fast, but it is not guaranteed to find the cheapest route.

### A*

A* combines the accumulated route cost and a lower-bound heuristic:

```text
f(n) = g(n) + h(n)
```

It is used as the recommended default cost-based search method.

## 6. Agent behaviour

### Goal-based agent

The goal-based agent treats requested destinations as hard goals. It removes destinations that fail the eligibility rules, finds a route visiting the remaining destinations, and selects the least expensive hotel in each city.

### Utility-based agent

The utility-based agent compares possible destination orders and scores them using:

- Budget fit: 45%
- Time fit: 25%
- Preference satisfaction: 20%
- Route simplicity: 10%

This makes the decision process visible instead of returning an unexplained answer.

## 7. GUI demonstration flow

1. Select the origin and destinations.
2. Set budget, days, and travelers.
3. Select interest categories.
4. Choose the agent and search algorithm.
5. Generate the itinerary.
6. Use `Compare Algorithms` with one destination to compare all five searches.
7. Use `View Knowledge Base` to show the stored facts.
8. Save the result as a text file.

## 8. Scope and assumptions

The project uses a local sample knowledge base rather than live booking or map APIs. Costs and travel times are approximate educational data. The design can later be extended with a database, live APIs, weather, dates, and a map interface.

## 9. Viva explanation

> The system uses a graph-based knowledge base and classical search algorithms to create an explainable travel plan. Hard constraints are handled by eligibility rules. A goal-based agent focuses on satisfying destination goals, while a utility-based agent balances cost, time, preferences, and route simplicity. The GUI exposes the inputs and displays the route, cost breakdown, selected activities, and reasoning so that the result can be evaluated.
