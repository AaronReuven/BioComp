import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
import time
import tkinter.messagebox as messagebox

from ex2 import GeneticAlgorithm, MagicSquareProblem  # assumes EX2.py is in the same folder


class MagicSquareApp:
    def __init__(self, master):
        self.master = master
        master.title("Magic Square Genetic Algorithm")

        self.running = False
        self.start_time = None

        self.main_frame = tk.Frame(master, padx=10, pady=10)
        self.main_frame.pack()

        self.square_frame = tk.Frame(self.main_frame)
        self.square_frame.grid(row=0, column=0, padx=10, sticky="nw")

        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.grid(row=0, column=1, padx=10, sticky="ne")

        row_counter = 0

        tk.Label(self.control_frame, text="Square size (N):").grid(row=row_counter, column=0, sticky="w")
        self.entry_n = tk.Entry(self.control_frame, width=8)
        self.entry_n.insert(0, "5")
        self.entry_n.grid(row=row_counter, column=1, sticky="w")
        row_counter += 1

        tk.Label(self.control_frame, text="Generations:").grid(row=row_counter, column=0, sticky="w")
        self.entry_gen = tk.Entry(self.control_frame, width=8)
        self.entry_gen.insert(0, "500")
        self.entry_gen.grid(row=row_counter, column=1, sticky="w")
        row_counter += 1

        # Checkbox: Run until fitness == 0
        self.var_run_until_solved = tk.BooleanVar(value=False)
        self.check_until = tk.Checkbutton(
            self.control_frame, text="Run until solved", variable=self.var_run_until_solved
        )
        self.check_until.grid(row=row_counter, column=0, columnspan=2, sticky="w")
        row_counter += 1

        tk.Label(self.control_frame, text="Mutation rate:").grid(row=row_counter, column=0, sticky="w")
        self.entry_mut = tk.Entry(self.control_frame, width=8)
        self.entry_mut.insert(0, "0.05")
        self.entry_mut.grid(row=row_counter, column=1, sticky="w")
        row_counter += 1

        tk.Label(self.control_frame, text="Elitism (0–N):").grid(row=row_counter, column=0, sticky="w")
        self.entry_elite = tk.Entry(self.control_frame, width=8)
        self.entry_elite.insert(0, "2")
        self.entry_elite.grid(row=row_counter, column=1, sticky="w")
        row_counter += 1

        tk.Label(self.control_frame, text="Population size:").grid(row=row_counter, column=0, sticky="w")
        self.entry_pop = tk.Entry(self.control_frame, width=8)
        self.entry_pop.insert(0, "100")
        self.entry_pop.grid(row=row_counter, column=1, sticky="w")
        row_counter += 1

        tk.Label(self.control_frame, text="Learning variant:").grid(row=row_counter, column=0, sticky="w")
        self.variant_var = tk.StringVar(value="none")
        radio_none = tk.Radiobutton(self.control_frame, text="None", variable=self.variant_var, value="none")
        radio_lamarck = tk.Radiobutton(self.control_frame, text="Lamarkian", variable=self.variant_var, value="lamarkian")
        radio_darwin = tk.Radiobutton(self.control_frame, text="Darwinian", variable=self.variant_var, value="darwinian")
        radio_none.grid(row=row_counter, column=1, sticky="w")
        row_counter += 1
        radio_lamarck.grid(row=row_counter, column=0, sticky="w")
        radio_darwin.grid(row=row_counter, column=1, sticky="w")
        row_counter += 1

        tk.Label(self.control_frame, text="Square type:").grid(row=row_counter, column=0, sticky="w")
        self.square_type_var = tk.StringVar(value="standard")
        radio_standard = tk.Radiobutton(self.control_frame, text="Standard", variable=self.square_type_var, value="standard")
        radio_perfect = tk.Radiobutton(self.control_frame, text="Most Perfect", variable=self.square_type_var, value="most_perfect")
        radio_standard.grid(row=row_counter, column=1, sticky="w")
        row_counter += 1
        radio_perfect.grid(row=row_counter, column=0, sticky="w")
        row_counter += 1


        tk.Label(self.control_frame, text="Time:").grid(row=row_counter, column=0, sticky="w")
        self.time_label = tk.Label(self.control_frame, text="00:00:00")
        self.time_label.grid(row=row_counter, column=1, sticky="w")
        row_counter += 1

        # Current generation label
        tk.Label(self.control_frame, text="Current gen:").grid(row=row_counter, column=0, sticky="w")
        self.gen_label = tk.Label(self.control_frame, text="0")
        self.gen_label.grid(row=row_counter, column=1, sticky="w")
        row_counter += 1

        self.buttons_frame = tk.Frame(self.control_frame)
        self.buttons_frame.grid(row=row_counter, column=0, columnspan=2, pady=5)
        row_counter += 1

        self.run_button = tk.Button(self.buttons_frame, text="Run", width=8, command=self.run_algorithm)
        self.run_button.pack(side="left", padx=2)
        self.stop_button = tk.Button(self.buttons_frame, text="Stop", width=8, command=self.stop_algorithm)
        self.stop_button.pack(side="left", padx=2)
        self.reset_button = tk.Button(self.buttons_frame, text="Reset", width=8, command=self.reset_ui)
        self.reset_button.pack(side="left", padx=2)

        self.labels = []

    def clear_display(self):
        for row in self.labels:
            for lbl in row:
                lbl.destroy()
        self.labels = []

    def update_square_display(self, square_2d, solved=False):
        """
        Draw a grid of Labels showing the current best square.
        If solved=True, color them green.
        """
        self.clear_display()
        n = square_2d.shape[0]
        for i in range(n):
            row_labels = []
            for j in range(n):
                val = square_2d[i, j]
                color = "green" if solved else "black"
                lbl = tk.Label(
                    self.square_frame,
                    text=str(val),
                    width=4,
                    height=2,
                    borderwidth=1,
                    relief="solid",
                    font=("Courier", 12),
                    fg=color
                )
                lbl.grid(row=i, column=j, padx=1, pady=1)
                row_labels.append(lbl)
            self.labels.append(row_labels)

    def run_algorithm(self):
        try:
            self.running = True
            self.clear_display()
            self.start_time = time.time()
            self.update_time_label()  # start updating elapsed time

            # Read parameters from UI
            n = int(self.entry_n.get())
            generations = int(self.entry_gen.get())
            mutation_rate = float(self.entry_mut.get())
            elitism = int(self.entry_elite.get())
            pop_size = int(self.entry_pop.get())
            square_mode = self.square_type_var.get()
            variant = self.variant_var.get()
            run_until_solved = self.var_run_until_solved.get()

            if variant == "lamarkian":
                learning_type = "lamarkian"
            elif variant == "darwinian":
                learning_type = "darwinian"
            else:
                learning_type = None

            mode = "most_perfect" if square_mode == "most_perfect" else "standard"

            # Build the GA
            ga = GeneticAlgorithm(
                MagicSquareProblem,
                problem_args={"size": n, "mode": mode},
                mutation_rate=mutation_rate,
                elitism=elitism,
                learning_type=learning_type,
                learning_cap=n,      # local search “steps” = N
                pop_size=pop_size,
                seed=42
            )

            best_fitness = float("inf")
            best_indiv = None

            generation_count = 0
            while self.running:
                if (not run_until_solved) and (generation_count >= generations):
                    break

                ga.population = ga.learning_step(ga.population)
                # 2) Next generation
                ga.population = ga.generation_step(ga.population)
                generation_count += 1

                # Update current generation label
                self.gen_label.config(text=str(generation_count))

                curr = min(ga.population, key=lambda ind: ind.fitness)
                curr_fit = curr.fitness
                if curr_fit < best_fitness:
                    best_fitness = curr_fit
                    best_indiv = curr.copy()
                    arr2d = np.array(best_indiv.flat).reshape(n, n)
                    self.update_square_display(arr2d)

                self.master.update_idletasks()
                self.master.update()

                #Early stop if running until solved and found solution
                if run_until_solved and (best_fitness == 0):
                    break

            self.running = False
            if best_indiv:
                final_square = np.array(best_indiv.flat).reshape(n, n)
                fig, ax = plt.subplots()
                ax.set_title(f"Best Fitness = {best_fitness} after {generation_count} gens")
                ax.axis("off")
                tbl = ax.table(cellText=final_square.tolist(), loc="center", cellLoc="center")
                tbl.scale(1, 2)
                plt.show()
                messagebox.showinfo("Done", "GA has finished running!")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Error:", e)

    def update_time_label(self):
        """
        """
        if not self.running or self.start_time is None:
            return

        elapsed = int(time.time() - self.start_time)
        hrs = elapsed // 3600
        mins = (elapsed % 3600) // 60
        secs = elapsed % 60
        self.time_label.config(text=f"{hrs:02d}:{mins:02d}:{secs:02d}")
        # Schedule next update in 500 ms
        self.master.after(500, self.update_time_label)

    def stop_algorithm(self):
        self.running = False

    def reset_ui(self):
        self.running = False
        self.clear_display()
        self.entry_n.delete(0, tk.END)
        self.entry_n.insert(0, "5")
        self.entry_gen.delete(0, tk.END)
        self.entry_gen.insert(0, "500")
        self.entry_mut.delete(0, tk.END)
        self.entry_mut.insert(0, "0.05")
        self.entry_elite.delete(0, tk.END)
        self.entry_elite.insert(0, "2")
        self.entry_pop.delete(0, tk.END)
        self.entry_pop.insert(0, "100")
        self.variant_var.set("none")
        self.square_type_var.set("standard")
        self.var_run_until_solved.set(False)
        self.time_label.config(text="00:00:00")
        self.gen_label.config(text="0")
        self.start_time = None


if __name__ == "__main__":
    root = tk.Tk()
    app = MagicSquareApp(root)
    root.mainloop()