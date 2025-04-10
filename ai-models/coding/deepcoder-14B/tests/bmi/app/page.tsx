'use client'

import { useState } from 'react';

export default function Home() {
  // State hooks for managing input values and results.
  const [weight, setWeight] = useState('');
  const [height, setHeight] = useState('');
  const [bmi, setBmi] = useState(null);
  const [category, setCategory] = useState('');

  // Function to determine health tips based on BMI category.
  const getHealthTips = (bmiCategory) => {
    switch (bmiCategory) {
      case 'Underweight':
        return (
          <>
            <p>Consider incorporating nutrient-dense foods into your meals.</p>
            <p>Include lean proteins, whole grains, healthy fats, and frequent snacks.</p>
            <p>If underweight is a concern, consult a nutritionist for personalized advice.</p>
          </>
        );
      case 'Normal weight':
        return (
          <>
            <p>Keep up the good work with a balanced diet.</p>
            <p>Focus on a variety of fruits, vegetables, lean proteins, and whole grains.</p>
            <p>Maintain an active lifestyle for overall health and wellbeing.</p>
          </>
        );
      case 'Overweight':
        return (
          <>
            <p>Focus on reducing refined sugars and saturated fats.</p>
            <p>Increase your intake of vegetables, fruits, and lean proteins.</p>
            <p>Consider moderate exercise to support a healthy weight loss journey.</p>
          </>
        );
      case 'Obesity':
        return (
          <>
            <p>It may help to adopt a lower-calorie, nutrient-dense diet.</p>
            <p>Increase whole foods like vegetables and fruits while minimizing processed foods.</p>
            <p>Consult with health professionals for a tailored plan.</p>
          </>
        );
      default:
        return null;
    }
  };

  // Function to calculate BMI and determine the category.
  const calculateBMI = (event) => {
    event.preventDefault();

    // Ensure input values are provided and convert height from centimeters to meters.
    if (weight && height) {
      const heightInMeters = height / 100;
      const bmiValue = weight / (heightInMeters * heightInMeters);

      // Update state with BMI value (rounded to 2 decimals).
      setBmi(bmiValue.toFixed(2));

      // Determine BMI category based on standard ranges.
      if (bmiValue < 18.5) {
        setCategory('Underweight');
      } else if (bmiValue >= 18.5 && bmiValue < 25) {
        setCategory('Normal weight');
      } else if (bmiValue >= 25 && bmiValue < 30) {
        setCategory('Overweight');
      } else {
        setCategory('Obesity');
      }
    }
  };

  // Reset function to clear inputs and results.
  const resetCalculator = () => {
    setWeight('');
    setHeight('');
    setBmi(null);
    setCategory('');
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-[#f6d365] to-[#fda085] font-sans">
      <div className="bg-white p-8 rounded-lg shadow-md max-w-md w-full">
        <h1 className="text-center text-2xl font-bold text-gray-800 mb-6">BMI Calculator</h1>
        <form onSubmit={calculateBMI}>
          <div className="mb-4">
            <label htmlFor="weight" className="block mb-2 font-medium">
              Weight (kg):
            </label>
            <input
              type="number"
              id="weight"
              value={weight}
              onChange={(e) => setWeight(e.target.value)}
              placeholder="e.g. 70"
              required
              className="w-full p-3 text-base rounded border border-gray-300 outline-none transition duration-300 focus:border-[#fda085]"
            />
          </div>
          <div className="mb-4">
            <label htmlFor="height" className="block mb-2 font-medium">
              Height (cm):
            </label>
            <input
              type="number"
              id="height"
              value={height}
              onChange={(e) => setHeight(e.target.value)}
              placeholder="e.g. 170"
              required
              className="w-full p-3 text-base rounded border border-gray-300 outline-none transition duration-300 focus:border-[#fda085]"
            />
          </div>
          <div className="flex justify-center mt-4">
            <button
              type="submit"
              className="px-6 py-3 rounded bg-[#fda085] text-white mr-2 hover:bg-[#f6a07a] transition duration-300"
            >
              Calculate BMI
            </button>
            <button
              type="button"
              onClick={resetCalculator}
              className="px-6 py-3 rounded bg-gray-300 text-gray-800 hover:bg-gray-400 transition duration-300"
            >
              Reset
            </button>
          </div>
        </form>

        {bmi && (
          <div className="mt-6 text-center p-4 border-t border-gray-200">
            <h2 className="text-xl font-semibold">Your BMI is: {bmi}</h2>
            <p className="mt-2">Category: {category}</p>
            <div className="mt-4 text-left">
              <h3 className="text-lg font-bold">Health & Diet Tips:</h3>
              {getHealthTips(category)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

