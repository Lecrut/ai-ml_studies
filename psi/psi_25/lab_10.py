import numpy as np


class MembershipFunction:
    def __init__(self, name, points):
        self.name = name
        self.points = points

    def __call__(self, x):
        p = self.points
        if len(p) == 3:
            a, b, c = p
            return max(min((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)), 0)
        elif len(p) == 4:
            a, b, c, d = p
            return max(min((x - a) / (b - a + 1e-9), 1, (d - x) / (d - c + 1e-9)), 0)
        else:
            raise ValueError("Invalid membership shape")


class LinguisticVariable:
    def __init__(self, name, universe):
        self.name = name
        self.universe = universe
        self.terms = {}

    def membership_function(self, label, points):
        self.terms[label] = MembershipFunction(label, points)

    def fuzzify(self, value):
        return {label: mf(value) for label, mf in self.terms.items()}


class FuzzyRule:
    def __init__(self, antecedents, output_value):
        self.antecedents = antecedents
        self.output_value = output_value

    def evaluate(self, input_values):
        degrees = []
        for var, label in self.antecedents:
            degree = var.terms[label](input_values[var.name])
            degrees.append(degree)
        return min(degrees)


class ControlSystem:
    def __init__(self):
        self.rules = []

    def add_rule(self, rule):
        self.rules.append(rule)


class ControlSystemSimulation:
    def __init__(self, system):
        self.system = system
        self.inputs = {}

    def input(self, **kwargs):
        self.inputs.update(kwargs)

    def compute(self):
        num = 0
        den = 0
        for rule in self.system.rules:
            weight = rule.evaluate(self.inputs)
            num += weight * rule.output_value
            den += weight
        self.output = num / den if den != 0 else 0

    def output_value(self):
        return self.output


def custom_fuzzy_tip(food_quality, service_quality):
    food = LinguisticVariable('food', np.linspace(0, 10, 100))
    food.membership_function('bad', [0, 0, 5])
    food.membership_function('average', [0, 5, 10])
    food.membership_function('good', [5, 10, 10])

    service = LinguisticVariable('service', np.linspace(0, 10, 100))
    service.membership_function('poor', [0, 0, 5])
    service.membership_function('ok', [0, 5, 10])
    service.membership_function('excellent', [5, 10, 10])

    rules = [
        FuzzyRule([(food, 'bad'), (service, 'poor')], 2),
        FuzzyRule([(food, 'bad'), (service, 'ok')], 5),
        FuzzyRule([(food, 'average'), (service, 'ok')], 8),
        FuzzyRule([(food, 'good'), (service, 'ok')], 10),
        FuzzyRule([(food, 'good'), (service, 'excellent')], 13),
        FuzzyRule([(food, 'average'), (service, 'excellent')], 12),
    ]

    system = ControlSystem()
    for rule in rules:
        system.add_rule(rule)

    sim = ControlSystemSimulation(system)
    sim.input(food=food_quality, service=service_quality)
    sim.compute()
    return sim.output_value()


if __name__ == '__main__':
    for i in range(11):
        for j in range(11):
            print(f"{i}, {j}: Własny system: {custom_fuzzy_tip(i, j):.2f}%")


