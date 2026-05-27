class PlotStyle:
    def __init__(self):
        self.facecolor = "whitesmoke"
        self.label_size = 14
        self.tick_size = 11
        self.dpi = 300
        
    def apply(self, fig, ax):
        ax.set_facecolor(self.facecolor)
        ax.tick_params(labelsize=self.tick_size)
        ax.set_axisbelow(True)
        ax.grid(True, color="black", linewidth=0.8, alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(bbox_to_anchor=(0, 1.25), loc='upper left')
        fig.tight_layout() 
    
    def get_rmae_label_name(self):
        return "Santykinė Vidutinė absoliuti paklaida\n(paklaida / bazinė paklaida)"
    
    def get_mae_label_name(self):
        return "Vidutinė absoliuti paklaida \n(EUR/MWh)"
    
    def get_rmse_label_name(self):
        return "Vid. kvad. paklaidos šaknis \n(EUR/MWh)"