import numpy as np
import math
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
input_hidden_weights = np.array([
    [-3.56916605112348e-002, -4.49475474039075e-001, 1.02362056650210e+000, 1.20129622423909e+000, 9.86318528634453e-001, 1.19528497890791e+000, -2.50743661436522e-001, 2.03656211229257e-001, -9.93868039614448e-002, 1.58980945517026e+000, -3.38545102844541e-001],
    [-3.11309891395907e-001, -2.90313981645813e-001, 7.80662477664456e-001, 4.59008158981510e-001, 7.04435676198771e-001, 6.80200455154486e-001, -2.13473359635676e-001, 3.04373226651372e-001, -2.77819603261783e-001, 1.24051880812656e+000, -1.42251410718079e-001],
    [1.98687927469521e-001, -1.14081735884590e-001, 1.40678206939451e+000, -9.69534163781797e-001, 2.95939530903593e-001, 6.87632973527515e-001, -4.89520240536631e-002, -2.26594796358661e-001, -3.81881325596283e-001, -1.37759350712363e-001, -3.98934101127756e-001],
    [-1.22598478061490e-001, 9.31863347269745e-002, -1.14212359416366e+000, 4.33224933906641e-001, -3.40197793431289e-001, -5.60024102319396e-001, -1.98323335549834e-001, 5.79319919450861e-002, -3.14689808766757e-001, 5.69482278108653e-001, -2.01378872498092e-001],
    [4.81148311970736e-002, 1.01877036759669e-001, 5.12708263760972e-001, 6.21595040428088e-001, 5.12473611472801e-001, 6.21314907496472e-001, -2.37716483807916e-001, 1.20546938171656e-001, -2.28313861467419e-002, 4.89281166324048e-001, -1.69441173349962e-001],
    [1.05565580825562e-001, -9.50565020920953e-002, 1.10864622705161e+000, -4.33710353952155e-001, 4.24588367585726e-001, 6.09198457936821e-001, -2.61957314373860e-001, -5.33184796598071e-001, -4.50860832036382e-001, 4.45402698795425e-002, -2.47862281678563e-001],
    [-1.19379940597319e-001, 2.62250981504089e-001, 1.62812856999600e+000, 1.54908103678298e+000, 1.55808733456540e+000, 1.58479570978760e+000, -8.59722245154669e-001, 4.55160684019764e-001, -4.02961326241336e-003, 2.67440303324318e+000, -6.56373998604649e-001],
    [2.08562282355245e-001, 1.88763278943152e-001, 6.67029065297284e-001, 1.00786668928565e+000, 7.48060020507218e-001, 7.87358106470000e-001, -2.76191476292663e-001, -4.46914561676044e-001, 3.61944473332107e-001, 8.67100950547849e-001, -2.93576287777477e-003]
])
hidden_bias = np.array([1.01779321685842e+000, 9.45292872093323e-001, -1.21939601784853e+000, -5.68964627307591e-002, 1.87086793833294e-001, -1.48144707188089e+000, 1.63452747275218e+000, 4.01140622410941e-001])
hidden_output_wts = np.array([
    [-8.35109501180963e-001, 9.82703944800878e-001, 1.03021662690211e+000, -1.08819171572418e+000, 2.33004207021384e-001, 4.82287597646489e-001, 3.93041169712753e-001, 1.68758553854008e-002]
])
output_bias = np.array([-3.08560834273878e-001])
max_input = np.array([5.24131000000000e+005, 4.87383156787000e+010, 1.46990000000000e+004, 1.48500000000000e+004, 1.44650000000000e+004, 1.49785000000000e+004])
min_input = np.array([6.41900000000000e+003, 2.43944490500000e+008, 1.14570000000000e+002, 1.13000000000000e+002, 1.12070000000000e+002, 1.16350000000000e+002])
max_target = np.array([1.46590000000000e+004])
min_target = np.array([1.13060000000000e+002])
mean_inputs = np.array([5.60076332574032e+004, 4.99724978761890e+009, 3.93687266514806e+003, 3.94125494305239e+003, 3.87518446469248e+003, 3.99499184510251e+003])
def logistic(x):
    return 1 / (1 + math.exp(-x)) if -100 < x < 100 else (1 if x > 100 else 0)

def scale_inputs(inputs):
    scaled = np.zeros_like(inputs)
    for i in range(6):
        delta = (1 - 0) / (max_input[i] - min_input[i])
        scaled[i] = 0 - delta * min_input[i] + delta * inputs[i]
    return scaled
def unscale_target(output):
    delta = (1 - 0) / (max_target[0] - min_target[0])
    return (output[0] - 0 + delta * min_target[0]) / delta
def compute_network(inputs):
    hidden = np.dot(input_hidden_weights, inputs) + hidden_bias
    hidden = np.array([logistic(x) for x in hidden])
    output = np.dot(hidden_output_wts, hidden) + output_bias
    return output
class StockPriceApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Прогноз цены закрытия акций")
        self.master.geometry("600x550")
        self.main_frame = ttk.Frame(master, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(self.main_frame, text="Прогнозирование цены закрытия акций",  font=("Helvetica", 14, "bold")).grid(column=0, row=0, columnspan=2, pady=10)
        input_frame = ttk.LabelFrame(self.main_frame, text="Входные параметры")
        input_frame.grid(column=0, row=1, padx=10, pady=10, sticky="nsew")
        result_frame = ttk.LabelFrame(self.main_frame, text="Результат")
        result_frame.grid(column=0, row=2, padx=10, pady=10, sticky="nsew")
       self.entries = {}
      ttk.Label(input_frame, text="Количество сделок:").grid(column=0, row=0, padx=5, pady=5, sticky="w")
        self.entries["kolvosdelok"] = ttk.Entry(input_frame, width=30)
        self.entries["kolvosdelok"].grid(column=1, row=0, padx=5, pady=5) 
        ttk.Label(input_frame, text="Объем:").grid(column=0, row=1, padx=5, pady=5, sticky="w")
        self.entries["obem"] = ttk.Entry(input_frame, width=30)
        self.entries["obem"].grid(column=1, row=1, padx=5, pady=5) 
        ttk.Label(input_frame, text="Средняя цена:").grid(column=0, row=2, padx=5, pady=5, sticky="w")
        self.entries["srznchprice"] = ttk.Entry(input_frame, width=30)
        self.entries["srznchprice"].grid(column=1, row=2, padx=5, pady=5) 
        ttk.Label(input_frame, text="Цена открытия:").grid(column=0, row=3, padx=5, pady=5, sticky="w")
        self.entries["open"] = ttk.Entry(input_frame, width=30)
        self.entries["open"].grid(column=1, row=3, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Минимум:").grid(column=0, row=4, padx=5, pady=5, sticky="w")
        self.entries["minimum"] = ttk.Entry(input_frame, width=30)
        self.entries["minimum"].grid(column=1, row=4, padx=5, pady=5) 
        ttk.Label(input_frame, text="Максимум:").grid(column=0, row=5, padx=5, pady=5, sticky="w")
        self.entries["maximum"] = ttk.Entry(input_frame, width=30)
        self.entries["maximum"].grid(column=1, row=5, padx=5, pady=5) 
        ttk.Label(input_frame, text="Инструмент:").grid(column=0, row=6, padx=5, pady=5, sticky="w")
        self.instrument_var = tk.StringVar()
        self.instrument_dropdown = ttk.Combobox(input_frame, textvariable=self.instrument_var, width=27)
        self.instrument_dropdown['values'] = ('GAZP', 'MGNT', 'MTSS', 'PLZL', 'SBER')
        self.instrument_dropdown.current(0)  
        self.instrument_dropdown.grid(column=1, row=6, padx=5, pady=5) 
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(column=0, row=7, columnspan=2, pady=10) 
        ttk.Button(button_frame, text="Рассчитать", command=self.calculate).grid(column=0, row=0, padx=10)
        ttk.Button(button_frame, text="Очистить", command=self.clear_form).grid(column=1, row=0, padx=10) 
        self.result_text = tk.Text(result_frame, height=5, width=50)
        self.result_text.grid(column=0, row=0, padx=10, pady=10)
        self.result_text.config(state="disabled") 
        info_frame = ttk.Frame(self.main_frame)
        info_frame.grid(column=0, row=3, padx=10, pady=5, sticky="w")
        ttk.Label(info_frame, text="* Пустые поля будут заменены средними значениями").grid(column=0, row=0, sticky="w") 
        hint_frame = ttk.Frame(self.main_frame)
        hint_frame.grid(column=0, row=4, padx=10, pady=5, sticky="w")
        ttk.Label(hint_frame, text="Средние значения:").grid(column=0, row=0, sticky="w", pady=2)
        ttk.Label(hint_frame, text=f"Кол-во сделок: {mean_inputs[0]:.2f}").grid(column=0, row=1, sticky="w")
        ttk.Label(hint_frame, text=f"Объем: {mean_inputs[1]:.2f}").grid(column=0, row=2, sticky="w")
        ttk.Label(hint_frame, text=f"Ср. цена: {mean_inputs[2]:.2f}").grid(column=0, row=3, sticky="w")
        ttk.Label(hint_frame, text=f"Открытие: {mean_inputs[3]:.2f}").grid(column=0, row=4, sticky="w")
        ttk.Label(hint_frame, text=f"Минимум: {mean_inputs[4]:.2f}").grid(column=0, row=5, sticky="w")
        ttk.Label(hint_frame, text=f"Максимум: {mean_inputs[5]:.2f}").grid(column=0, row=6, sticky="w") 
    def clear_form(self):
           for entry in self.entries.values():
            entry.delete(0, tk.END)
        
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state="disabled") 
    def calculate(self):
        try:
            inputs = np.zeros(11) 
            field_names = ["kolvosdelok", "obem", "srznchprice", "open", "minimum", "maximum"] 
            for i, field in enumerate(field_names):
                value = self.entries[field].get().strip() 
                if not value:
                    inputs[i] = mean_inputs[i]
                else:
                    try:
                        inputs[i] = float(value)
                    except ValueError:
                        messagebox.showerror("Ошибка", f"Некорректное значение в поле '{field}'")
                        return 
            instrument_name = self.instrument_var.get()
            instrument_index = ["GAZP", "MGNT", "MTSS", "PLZL", "SBER"].index(instrument_name) 
            inputs[6:11] = 0
            inputs[6 + instrument_index] = 1.0 
            scaled_inputs = scale_inputs(inputs[:6])
            full_inputs = np.concatenate((scaled_inputs, inputs[6:]))
            output = compute_network(full_inputs)
            result = unscale_target(output) 
            self.result_text.config(state="normal")
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Инструмент: {instrument_name}\n\n")
            self.result_text.insert(tk.END, f"Прогнозируемая цена закрытия: {result:.2f}")
            self.result_text.config(state="disabled") 
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при расчете: {str(e)}") 
def main():
    root = tk.Tk()
    app = StockPriceApp(root)
    root.mainloop()
if __name__ == "__main__":
    main()





