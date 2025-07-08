class Discretizer:
    def __init__(self, continuous_min, continuous_max, discrete_min, discrete_max):
        """
        Initialize the Discretizer with the range of continuous and discrete values.

        Parameters:
        - continuous_min: The minimum value of the continuous range.
        - continuous_max: The maximum value of the continuous range.
        - discrete_min: The minimum value of the discrete range.
        - discrete_max: The maximum value of the discrete range.
        """
        self.continuous_min = continuous_min
        self.continuous_max = continuous_max
        self.discrete_min = discrete_min
        self.discrete_max = discrete_max

    def discretize(self, values):
        """
        Discretize a list of continuous values to their corresponding discrete values.

        Parameters:
        - values: A list of continuous values to be discretized.

        Returns:
        - A list of corresponding discrete values.
        """
        if not isinstance(values, list):
            raise TypeError("Input should be a list of values.")

        discrete_values = []
        for value in values:
            # Ensure the value is within the continuous range
            if value < self.continuous_min or value > self.continuous_max:
                raise ValueError(
                    f"Value {value} is outside the range [{self.continuous_min}, {self.continuous_max}]."
                )

            # Map the continuous value to the discrete range
            normalized_value = (value - self.continuous_min) / (
                self.continuous_max - self.continuous_min
            )
            discrete_value = self.discrete_min + normalized_value * (
                self.discrete_max - self.discrete_min
            )

            # Round to the nearest integer for discrete value
            discrete_values.append(round(discrete_value))

        return discrete_values
