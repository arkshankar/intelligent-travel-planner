"""Tkinter desktop interface for the intelligent travel-planning agent."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from travel_agent import TravelKnowledgeBase, build_agent, choose_route, make_request


class TravelPlannerApp(tk.Tk):
    """A form-based desktop dashboard around the travel-planning agents."""

    COLORS = {
        "navy": "#183b56",
        "blue": "#2878a8",
        "page": "#edf2f7",
        "card": "#ffffff",
        "text": "#1f2933",
        "muted": "#52606d",
        "green": "#2f855a",
        "orange": "#c05621",
        "border": "#d9e2ec",
    }

    INTERESTS = ("history", "beach", "nature", "shopping", "food")
    ALGORITHMS = {
        "A* (cost + heuristic)": "astar",
        "Uniform Cost Search": "ucs",
        "Greedy Best-First": "greedy",
        "Breadth-First Search": "bfs",
        "Depth-First Search": "dfs",
    }
    AGENTS = {
        "Utility-based agent": "utility",
        "Goal-based agent": "goal",
    }

    def __init__(self, knowledge_base: TravelKnowledgeBase | None = None) -> None:
        super().__init__()
        self.kb = knowledge_base or TravelKnowledgeBase()
        self.last_itinerary = None
        self.title("Intelligent Travel Planning Agent")
        self.geometry("1180x760")
        self.minsize(980, 580)
        self.configure(bg=self.COLORS["page"])
        self._configure_styles()
        self._create_variables()
        self._build_interface()
        self._load_defaults()

    def _create_variables(self) -> None:
        self.origin_var = tk.StringVar()
        self.budget_var = tk.StringVar(value="25000")
        self.days_var = tk.StringVar(value="6")
        self.travelers_var = tk.StringVar(value="1")
        self.agent_var = tk.StringVar(value="Utility-based agent")
        self.algorithm_var = tk.StringVar(value="A* (cost + heuristic)")
        self.interest_vars = {interest: tk.BooleanVar(value=False) for interest in self.INTERESTS}
        self.status_var = tk.StringVar(value="Choose your trip details and click Generate Itinerary.")
        self.destination_count_var = tk.StringVar(value="0 destinations selected")
        self.algorithm_info_var = tk.StringVar(value="A* balances route cost and a heuristic estimate.")
        self.route_summary_var = tk.StringVar(value="Not generated")
        self.cost_summary_var = tk.StringVar(value="INR -")
        self.time_summary_var = tk.StringVar(value="- h")
        self.budget_summary_var = tk.StringVar(value="Not calculated")

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=self.COLORS["page"])
        style.configure("TLabel", background=self.COLORS["page"], foreground=self.COLORS["text"], font=("Segoe UI", 10))
        style.configure("Muted.TLabel", foreground=self.COLORS["muted"], font=("Segoe UI", 9))
        style.configure("TLabelframe", background=self.COLORS["page"], bordercolor=self.COLORS["border"])
        style.configure("TLabelframe.Label", background=self.COLORS["page"], foreground=self.COLORS["navy"], font=("Segoe UI", 10, "bold"))
        style.configure("TButton", font=("Segoe UI", 9), padding=(9, 6))
        style.configure("Accent.TButton", background=self.COLORS["blue"], foreground="white", font=("Segoe UI", 9, "bold"))
        style.map("Accent.TButton", background=[("active", self.COLORS["navy"])])
        style.configure("TCheckbutton", background=self.COLORS["page"], foreground=self.COLORS["text"])
        style.configure("TCombobox", padding=4)
        style.configure("TEntry", padding=4)
        style.configure("Horizontal.TProgressbar", troughcolor="#d9e2ec", background=self.COLORS["green"])

    def _build_interface(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        header = tk.Frame(self, bg=self.COLORS["navy"], padx=20, pady=12)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.columnconfigure(0, weight=1)
        title_block = tk.Frame(header, bg=self.COLORS["navy"])
        title_block.grid(row=0, column=0, sticky="w")
        tk.Label(
            title_block,
            text="Intelligent Travel Planning Agent",
            font=("Segoe UI", 18, "bold"),
            fg="white",
            bg=self.COLORS["navy"],
        ).pack(anchor="w")
        tk.Label(
            title_block,
            text="Semester project | Knowledge base + search algorithms + explainable planning",
            font=("Segoe UI", 9),
            fg="#d8e8f2",
            bg=self.COLORS["navy"],
        ).pack(anchor="w", pady=(2, 0))
        ttk.Button(header, text="About project", command=self.show_about).grid(row=0, column=1, sticky="e")

        # Scrollable container for the left inputs
        left_container = ttk.Frame(self)
        left_container.grid(row=1, column=0, sticky="nsew", padx=(14, 6), pady=(8, 12))
        left_container.rowconfigure(0, weight=1)
        left_container.columnconfigure(0, weight=1)

        input_canvas = tk.Canvas(left_container, bg=self.COLORS["page"], highlightthickness=0, borderwidth=0)
        input_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=input_canvas.yview)
        input_canvas.configure(yscrollcommand=input_scrollbar.set)

        input_panel = ttk.LabelFrame(input_canvas, text="Trip details", padding=12)
        panel_window = input_canvas.create_window((0, 0), window=input_panel, anchor="nw")

        def _on_panel_configure(event):
            input_canvas.configure(scrollregion=input_canvas.bbox("all"))

        def _on_canvas_configure(event):
            input_canvas.itemconfig(panel_window, width=event.width)

        input_panel.bind("<Configure>", _on_panel_configure)
        input_canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            # Support mouse scrolling over input pane
            if event.delta:
                input_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif event.num == 4:
                input_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                input_canvas.yview_scroll(1, "units")

        input_canvas.bind_all("<MouseWheel>", lambda e: self._handle_scroll(e, input_canvas), add="+")

        input_canvas.grid(row=0, column=0, sticky="nsew")
        input_scrollbar.grid(row=0, column=1, sticky="ns")
        input_panel.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(input_panel, text="Starting city").grid(row=row, column=0, sticky="w", pady=4)
        self.origin_combo = ttk.Combobox(
            input_panel,
            textvariable=self.origin_var,
            values=sorted(self.kb.cities),
            state="readonly",
            width=23,
        )
        self.origin_combo.grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        ttk.Label(input_panel, text="Destinations").grid(row=row, column=0, sticky="nw", pady=4)
        destination_frame = ttk.Frame(input_panel)
        destination_frame.grid(row=row, column=1, sticky="ew", pady=4)
        destination_frame.columnconfigure(0, weight=1)
        self.destination_list = tk.Listbox(
            destination_frame,
            selectmode=tk.MULTIPLE,
            height=4,
            exportselection=False,
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 10),
        )
        for city in sorted(self.kb.cities):
            self.destination_list.insert(tk.END, city)
        self.destination_list.grid(row=0, column=0, sticky="ew")
        destination_scroll = ttk.Scrollbar(destination_frame, orient="vertical", command=self.destination_list.yview)
        destination_scroll.grid(row=0, column=1, sticky="ns")
        self.destination_list.configure(yscrollcommand=destination_scroll.set)
        self.destination_list.bind("<<ListboxSelect>>", self._update_destination_count)
        ttk.Label(
            destination_frame,
            textvariable=self.destination_count_var,
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))

        row += 1
        ttk.Label(input_panel, text="Budget (INR)").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(input_panel, textvariable=self.budget_var, width=25).grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        ttk.Label(input_panel, text="Duration (days)").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(input_panel, textvariable=self.days_var, width=25).grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        ttk.Label(input_panel, text="Travelers").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(input_panel, textvariable=self.travelers_var, width=25).grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        ttk.Label(input_panel, text="Interests").grid(row=row, column=0, sticky="nw", pady=4)
        interests_frame = ttk.Frame(input_panel)
        interests_frame.grid(row=row, column=1, sticky="w", pady=2)
        for index, interest in enumerate(self.INTERESTS):
            ttk.Checkbutton(
                interests_frame,
                text=interest.title(),
                variable=self.interest_vars[interest],
            ).grid(row=index // 2, column=index % 2, sticky="w", padx=(0, 10), pady=1)

        row += 1
        ttk.Label(input_panel, text="Agent type").grid(row=row, column=0, sticky="w", pady=4)
        self.agent_combo = ttk.Combobox(
            input_panel,
            textvariable=self.agent_var,
            values=list(self.AGENTS),
            state="readonly",
            width=23,
        )
        self.agent_combo.grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        ttk.Label(input_panel, text="Route algorithm").grid(row=row, column=0, sticky="w", pady=4)
        self.algorithm_combo = ttk.Combobox(
            input_panel,
            textvariable=self.algorithm_var,
            values=list(self.ALGORITHMS),
            state="readonly",
            width=23,
        )
        self.algorithm_combo.grid(row=row, column=1, sticky="ew", pady=4)
        self.algorithm_combo.bind("<<ComboboxSelected>>", self._update_algorithm_info)
        ttk.Label(
            input_panel,
            textvariable=self.algorithm_info_var,
            style="Muted.TLabel",
            wraplength=245,
        ).grid(row=row + 1, column=1, sticky="w", pady=(0, 2))

        row += 2
        buttons = ttk.Frame(input_panel)
        buttons.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 4))
        for column in range(3):
            buttons.columnconfigure(column, weight=1)
        ttk.Button(buttons, text="Generate Itinerary", style="Accent.TButton", command=self.generate_itinerary).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(buttons, text="Compare Algorithms", command=self.compare_algorithms).grid(row=0, column=1, sticky="ew", padx=3)
        ttk.Button(buttons, text="Clear", command=self.clear_form).grid(row=0, column=2, sticky="ew", padx=(3, 0))

        row += 1
        ttk.Button(input_panel, text="View Knowledge Base", command=self.show_knowledge_base).grid(row=row, column=0, columnspan=2, sticky="ew", pady=3)

        output_panel = ttk.LabelFrame(self, text="Generated itinerary", padding=12)
        output_panel.grid(row=1, column=1, sticky="nsew", padx=(6, 14), pady=(8, 12))
        output_panel.rowconfigure(1, weight=1)
        output_panel.columnconfigure(0, weight=1)

        summary = ttk.Frame(output_panel)
        summary.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        summary.columnconfigure(0, weight=1)
        summary.columnconfigure(1, weight=1)
        summary.columnconfigure(2, weight=1)
        summary.columnconfigure(3, weight=1)
        self._create_metric(summary, 0, "ROUTE", self.route_summary_var)
        self._create_metric(summary, 1, "TOTAL COST", self.cost_summary_var)
        self._create_metric(summary, 2, "TRAVEL TIME", self.time_summary_var)
        self._create_metric(summary, 3, "BUDGET", self.budget_summary_var)
        self.budget_progress = ttk.Progressbar(summary, style="Horizontal.TProgressbar", maximum=100, value=0)
        self.budget_progress.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(8, 0))

        self.output = tk.Text(
            output_panel,
            wrap="word",
            state="disabled",
            padx=14,
            pady=14,
            font=("Segoe UI", 10),
            bg=self.COLORS["card"],
            fg=self.COLORS["text"],
            relief="solid",
            borderwidth=1,
        )
        self.output.grid(row=1, column=0, sticky="nsew")
        output_scroll = ttk.Scrollbar(output_panel, orient="vertical", command=self.output.yview)
        output_scroll.grid(row=1, column=1, sticky="ns")
        self.output.configure(yscrollcommand=output_scroll.set)
        output_buttons = ttk.Frame(output_panel)
        output_buttons.grid(row=2, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(output_buttons, text="Save Itinerary as Text File", command=self.save_itinerary).pack(side="right")

        status = ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w", padding=(10, 5))
        status.grid(row=2, column=0, columnspan=2, sticky="ew")

    def _handle_scroll(self, event, canvas: tk.Canvas) -> None:
        # Only scroll the input canvas if the mouse is currently hovering over it or its children
        x, y = self.winfo_pointerxy()
        widget_under_mouse = self.winfo_containing(x, y)
        if widget_under_mouse is not None:
            # Check if mouse is over the left panel / canvas
            cur = widget_under_mouse
            while cur is not None:
                if cur == canvas:
                    if event.delta:
                        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                    elif event.num == 4:
                        canvas.yview_scroll(-1, "units")
                    elif event.num == 5:
                        canvas.yview_scroll(1, "units")
                    break
                cur = getattr(cur, "master", None)

    def _create_metric(self, parent: ttk.Frame, column: int, title: str, variable: tk.StringVar) -> None:
        card = tk.Frame(
            parent,
            bg=self.COLORS["card"],
            highlightbackground=self.COLORS["border"],
            highlightthickness=1,
            padx=10,
            pady=7,
        )
        card.grid(row=0, column=column, sticky="ew", padx=3)
        tk.Label(card, text=title, bg=self.COLORS["card"], fg=self.COLORS["muted"], font=("Segoe UI", 8, "bold")).pack(anchor="w")
        tk.Label(card, textvariable=variable, bg=self.COLORS["card"], fg=self.COLORS["navy"], font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(2, 0))

    def _load_defaults(self) -> None:
        self.origin_var.set("Delhi")
        for city in ("Jaipur", "Agra"):
            index = sorted(self.kb.cities).index(city)
            self.destination_list.selection_set(index)
        self.interest_vars["history"].set(True)
        self.interest_vars["food"].set(True)
        self._update_destination_count()
        self._update_algorithm_info()
        self._reset_summary()
        self._set_output(
            "Welcome to the travel-planning dashboard.\n\n"
            "1. Select a starting city and at least one destination.\n"
            "2. Set your budget, duration, travelers, and interests.\n"
            "3. Choose an agent and route algorithm.\n"
            "4. Click Generate Itinerary to see the route, cost breakdown, and reasoning.\n\n"
            "Recommended demonstration: Utility-based agent + A*."
        )

    def _update_destination_count(self, _event=None) -> None:
        count = len(self.destination_list.curselection())
        label = "destination" if count == 1 else "destinations"
        self.destination_count_var.set(f"{count} {label} selected")

    def _update_algorithm_info(self, _event=None) -> None:
        descriptions = {
            "astar": "A* balances route cost and a heuristic estimate.",
            "ucs": "UCS guarantees the lowest transport-cost route in this graph.",
            "greedy": "Greedy chooses the next node using the heuristic only.",
            "bfs": "BFS explores by levels and prefers fewer connections.",
            "dfs": "DFS explores one branch deeply before backtracking.",
        }
        key = self.ALGORITHMS.get(self.algorithm_var.get(), "astar")
        self.algorithm_info_var.set(descriptions[key])

    def _reset_summary(self) -> None:
        self.route_summary_var.set("Not generated")
        self.cost_summary_var.set("INR -")
        self.time_summary_var.set("- h")
        self.budget_summary_var.set("Not calculated")
        self.budget_progress.configure(value=0)

    def _update_summary(self, itinerary) -> None:
        route = " -> ".join(itinerary.route.cities)
        self.route_summary_var.set(route if len(route) < 34 else f"{itinerary.route.cities[0]} -> ... -> {itinerary.route.cities[-1]}")
        self.cost_summary_var.set(f"INR {itinerary.total_cost:,.0f}")
        self.time_summary_var.set(f"{itinerary.total_hours:.1f} h")
        within_budget = itinerary.total_cost <= itinerary.request.budget
        self.budget_summary_var.set("Within budget" if within_budget else "Over budget")
        percentage = min(100, max(0, itinerary.total_cost / max(1, itinerary.request.budget) * 100))
        self.budget_progress.configure(value=percentage)

    def _set_output(self, value: str) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.insert("1.0", value)
        self.output.configure(state="disabled")

    def _selected_destinations(self) -> tuple[str, ...]:
        return tuple(self.destination_list.get(index) for index in self.destination_list.curselection())

    def generate_itinerary(self) -> None:
        try:
            destinations = self._selected_destinations()
            if not destinations:
                raise ValueError("Select at least one destination.")
            interests = tuple(interest for interest, variable in self.interest_vars.items() if variable.get())
            request = make_request(
                self.origin_var.get(),
                destinations,
                float(self.budget_var.get()),
                float(self.days_var.get()),
                interests,
                int(self.travelers_var.get()),
                self.ALGORITHMS[self.algorithm_var.get()],
            )
            agent = build_agent(self.AGENTS[self.agent_var.get()], self.kb)
            self.last_itinerary = agent.plan(request)
        except (KeyError, ValueError) as error:
            self.status_var.set("Could not create itinerary.")
            messagebox.showerror("Invalid trip details", str(error), parent=self)
            return
        self._set_output(self.last_itinerary.as_text())
        self._update_summary(self.last_itinerary)
        self.status_var.set("Itinerary generated successfully.")

    def clear_form(self) -> None:
        self.budget_var.set("25000")
        self.days_var.set("6")
        self.travelers_var.set("1")
        self.agent_var.set("Utility-based agent")
        self.algorithm_var.set("A* (cost + heuristic)")
        self.destination_list.selection_clear(0, tk.END)
        for variable in self.interest_vars.values():
            variable.set(False)
        self.last_itinerary = None
        self._update_destination_count()
        self._reset_summary()
        self._set_output("Form cleared. Enter your travel preferences and generate a new itinerary.")
        self.status_var.set("Ready for a new trip.")

    def save_itinerary(self) -> None:
        if self.last_itinerary is None:
            messagebox.showinfo("Nothing to save", "Generate an itinerary first.", parent=self)
            return
        filename = filedialog.asksaveasfilename(
            parent=self,
            title="Save itinerary",
            defaultextension=".txt",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
        )
        if not filename:
            return
        try:
            with open(filename, "w", encoding="utf-8") as file:
                file.write(self.last_itinerary.as_text())
        except OSError as error:
            messagebox.showerror("Save failed", str(error), parent=self)
            return
        self.status_var.set(f"Itinerary saved to {filename}")
        messagebox.showinfo("Saved", "The itinerary was saved successfully.", parent=self)

    def compare_algorithms(self) -> None:
        """Show how the five search strategies behave for one route."""
        destinations = self._selected_destinations()
        if len(destinations) != 1:
            messagebox.showinfo(
                "Compare one route at a time",
                "Select exactly one destination before comparing algorithms.",
                parent=self,
            )
            return
        origin = self.origin_var.get()
        goal = destinations[0]
        comparison = tk.Toplevel(self)
        comparison.title(f"Algorithm comparison: {origin} to {goal}")
        comparison.geometry("860x430")
        comparison.minsize(700, 350)
        comparison.columnconfigure(0, weight=1)
        comparison.rowconfigure(1, weight=1)
        ttk.Label(
            comparison,
            text=f"Comparing route strategies from {origin} to {goal}",
            font=("Segoe UI", 12, "bold"),
            padding=(12, 12),
        ).grid(row=0, column=0, sticky="w")
        table_frame = ttk.Frame(comparison, padding=(12, 0, 12, 12))
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        columns = ("algorithm", "route", "connections", "cost", "time")
        table = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)
        headings = {
            "algorithm": "Algorithm",
            "route": "Selected route",
            "connections": "Edges",
            "cost": "Transport cost",
            "time": "Travel time",
        }
        widths = {"algorithm": 175, "route": 300, "connections": 75, "cost": 125, "time": 110}
        for column in columns:
            table.heading(column, text=headings[column])
            table.column(column, width=widths[column], anchor="w")
        table.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=table.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        table.configure(yscrollcommand=scroll.set)
        for label, key in (
            ("Breadth-First Search", "bfs"),
            ("Depth-First Search", "dfs"),
            ("Uniform Cost Search", "ucs"),
            ("Greedy Best-First", "greedy"),
            ("A*", "astar"),
        ):
            try:
                route = choose_route(self.kb, origin, goal, key)
                table.insert(
                    "",
                    "end",
                    values=(
                        label,
                        " -> ".join(route.cities),
                        len(route.legs),
                        f"INR {route.cost:,.0f}",
                        f"{route.hours:.1f} h",
                    ),
                )
            except ValueError as error:
                table.insert("", "end", values=(label, str(error), "-", "-", "-"))
        ttk.Label(
            comparison,
            text="BFS/DFS demonstrate exploration; UCS and A* are cost-oriented; Greedy is heuristic-driven.",
            style="Muted.TLabel",
            padding=(12, 0, 12, 10),
        ).grid(row=2, column=0, sticky="w")

    def show_about(self) -> None:
        messagebox.showinfo(
            "About this semester project",
            "INTELLIGENT TRAVEL PLANNING AGENT\n\n"
            "This project models travel planning as a graph-search and decision-making problem.\n\n"
            "Core components:\n"
            "- Knowledge base of cities, attractions, hotels, and transport links\n"
            "- Eligibility rules based on interests, time, and budget\n"
            "- BFS, DFS, UCS, Greedy Best-First, and A* search\n"
            "- Goal-based and utility-based agents\n"
            "- Explainable itinerary with route and cost reasoning\n\n"
            "Built as a Python B.Tech semester project using Tkinter and the standard library.",
            parent=self,
        )

    def show_knowledge_base(self) -> None:
        window = tk.Toplevel(self)
        window.title("Travel Knowledge Base")
        window.geometry("850x560")
        window.minsize(650, 400)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        text = tk.Text(window, wrap="word", padx=14, pady=14, font=("Consolas", 10))
        text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(window, orient="vertical", command=text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scroll.set)
        lines = ["CITIES AND KNOWLEDGE BASE FACTS", "=" * 34]
        for city in sorted(self.kb.cities.values(), key=lambda item: item.name):
            lines.append(f"\n{city.name} ({city.country})")
            lines.append("  Attractions: " + ", ".join(f"{item.name} [{item.category}]" for item in city.attractions))
            lines.append("  Hotels: " + ", ".join(f"{item.name} INR {item.nightly_cost:g}/night" for item in city.hotels))
            lines.append("  Connections: " + ", ".join(f"{item.destination} by {item.mode}, {item.hours:g}h, INR {item.cost:g}" for item in self.kb.neighbors(city.name)))
        text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")


def launch_gui(knowledge_base: TravelKnowledgeBase | None = None) -> int:
    app = TravelPlannerApp(knowledge_base)
    app.mainloop()
    return 0

