import numpy as np
import matplotlib.pyplot as plt

class ProbabilityModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def CumulativeDensityFunction(self, time_range) -> np.array:
        if self.model_name == "Exponential":
            lambda0 = 0.01
            return 1 - np.exp(-lambda0*time_range)
        
        elif self.model_name == "Weibull":
            alpha = 1.5
            lambda0 = 200
            return 1 - np.exp(-(time_range/lambda0)**alpha)
        
        elif self.model_name == "Log-logistic":
            alpha = 200.0
            beta = 2.5
            return 1 - 1.0 / (1.0 + (time_range / alpha) ** beta)
        
        elif self.model_name == "Gompertz":
            B = 0.001
            C = 0.01
            return 1 - np.exp(- (B / C) * (np.exp(C * time_range) - 1.0))

    def SurvivalFunc(self, time_range) -> np.array:
        if self.model_name == "Exponential":
            lambda0 = 0.01
            return np.exp(-lambda0*time_range)

        elif self.model_name == "Weibull":
            alpha = 1.5
            lambda0 = 200
            return np.exp(-(time_range/lambda0)**alpha)

        elif self.model_name == "Log-logistic":
            alpha = 200.0
            beta = 2.5
            return 1.0 / (1.0 + (time_range / alpha) ** beta)
        
        elif self.model_name == "Gompertz":
            B = 0.001
            C = 0.01
            return np.exp(- (B / C) * (np.exp(C * time_range) - 1.0))
    
    def HazardFunc(self, time_range) -> np.array:
        if self.model_name == "Exponential":
            lambda0 = 0.01
            return np.array([lambda0 for i in time_range])
        
        elif self.model_name == "Weibull":
            alpha = 1.5
            lambda0 = 200
            return (alpha / lambda0) * (time_range / lambda0) ** (alpha-1)

        elif self.model_name == "Log-logistic":
            alpha = 200.0
            beta = 2.5
            numerator = (beta / alpha) * ( (time_range / alpha) ** (beta - 1) )
            denominator = 1.0 + ( (time_range / alpha) ** beta )
            return numerator / denominator
        
        elif self.model_name == "Gompertz":
            B = 0.001
            C = 0.01
            return B * np.exp(C * time_range)

    def draw(self):
        # time range
        time_range = np.array([i for i in range(365)])

        # CDF: Cumulative Density Function 
        CDF = self.CumulativeDensityFunction(time_range)
        plt.figure(figsize=(7,5))
        plt.plot(time_range, CDF)
        plt.title("CDF ("+str(self.model_name)+")")
        plt.xlabel("Running Time (d)")
        plt.ylabel("Failure probability")
        plt.grid(True, ls="--", alpha=0.6)
        plt.show()


        # Survival Function
        SF = self.SurvivalFunc(time_range)
        plt.figure(figsize=(7,5))
        plt.plot(time_range, SF)
        plt.title("Survival Function ("+str(self.model_name)+")")
        plt.xlabel("Running Time (d)")
        plt.ylabel("Survival probability")
        plt.grid(True, ls="--", alpha=0.6)
        plt.show()


        # Hazard Function
        HF = self.HazardFunc(time_range)
        plt.figure(figsize=(7,5))
        plt.plot(time_range, HF)
        plt.title("Hazard Function ("+str(self.model_name)+")")
        plt.xlabel("Running Time (d)")
        plt.ylabel("Survival probability at each time")
        plt.grid(True, ls="--", alpha=0.6)
        plt.show()



