# Intelligent Travel Planning Agent

This is a Python semester project that creates a travel plan based on:

- Starting city and destinations
- Budget
- Number of days
- Number of travellers
- Travel interests
- Route-search algorithm

The project stores sample information about cities, attractions, hotels, transport, travel time, and approximate costs. It then creates an itinerary with a route, activities, hotel choices, cost breakdown, and a short explanation.

## Requirements

- Python 3.10 or newer
- Windows, macOS, or Linux
- Tkinter (usually included with Python on Windows)

No external Python packages are needed.

## Download the project

After cloning the repository, open a terminal inside the project folder.

```text
git clone https://github.com/arkshankar/intelligent-travel-planner
cd intelligent-travel-planner
```

Replace `https://github.com/arkshankar/intelligent-travel-planner` with the repository link.

## Run the desktop app

The easiest way is:

```text
python main.py
```

On Windows, you can also double-click:

```text
run_gui.bat
```

The window lets you choose the starting city, destinations, budget, days, travellers, interests, agent, and search algorithm.

## How to use the app

1. Select a starting city.
2. Select one or more destinations. Hold `Ctrl` to select multiple cities.
3. Enter the budget, number of days, and number of travellers.
4. Select interests such as history, beach, nature, shopping, or food.
5. Choose an agent and route algorithm.
6. Click `Generate Itinerary`.

The app shows the selected route, travel details, hotel, activities, estimated cost, budget status, and reasoning.

The app also has buttons to:

- Compare the five route-search algorithms
- View the knowledge base
- Save the itinerary as a text file
- Read a short description of the project

## Run the command-line version

The GUI is the default. To use the command line, add `--cli`:

```text
python main.py --cli
```

Example:

```text
python main.py --cli --from Delhi --to Jaipur Agra --budget 25000 --days 6 --interests history,food --agent utility --algorithm astar
```

To print the stored cities and connections:

```text
python main.py --show-kb
```

## Available choices

### Cities

Delhi, Jaipur, Agra, Mumbai, Goa, and Bengaluru.

### Interests

`history`, `beach`, `nature`, `shopping`, and `food`.

### Agents

- `goal` - focuses on reaching the selected destinations.
- `utility` - balances budget, time, interests, and route simplicity.

### Algorithms

- `bfs` - Breadth-First Search
- `dfs` - Depth-First Search
- `ucs` - Uniform Cost Search
- `greedy` - Greedy Best-First Search
- `astar` - A* Search

## Run the tests

```text
python -m unittest -v
```

## Project files

```text
main.py                    Starts the GUI or command-line version
gui.py                     Tkinter desktop interface
travel_agent.py            Knowledge base, algorithms, and agents
test_travel_agent.py       Automated tests
PROJECT_DOCUMENTATION.md   Project explanation and viva notes
run_gui.bat                Windows one-click launcher
```

## Notes

The project uses sample travel data stored in `travel_agent.py`. It does not use live maps, hotel websites, or booking services. The costs and travel times are approximate and are meant for demonstrating the AI concepts.
