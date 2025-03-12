"use client"

import { useState } from 'react';

export default function Calculator() {
  const [currentInput, setCurrentInput] = useState('');

  const handleNumber = (number: string) => {
    setCurrentInput(prev => prev + number);
  };

  const handleOperator = (operator: string) => {
    if (['+', '-', '*', '/'].includes(operator)) {
      const lastChar = currentInput.slice(-1);
      if (lastChar && ['+', '-', '*', '/'].includes(lastChar)) {
        // replace last operator
        setCurrentInput(currentInput.slice(0, -1) + operator);
      } else {
        setCurrentInput(prev => prev + operator);
      }
    }
  };

  const handleDecimal = () => {
    if (!currentInput.includes('.')) {
      setCurrentInput(prev => prev + '.');
    }
  };

  const handleClear = () => {
    setCurrentInput('');
  };

  const handleEquals = () => {
    try {
      const calculated = eval(currentInput);
      setCurrentInput(calculated.toString());
    } catch (error) {
      setCurrentInput('Error');
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <div className="bg-white shadow-md p-4 rounded-lg w-full max-w-md">
        <div className="text-right text-2xl mb-4">
          {currentInput || '0'}
        </div>
        <div className="grid grid-cols-4 gap-2">
          <button
            onClick={handleClear}
            className="bg-blue-500 text-white p-4 hover:bg-blue-600"
          >
            AC
          </button>
          <button
            onClick={() => handleOperator('/')}
            className="bg-blue-500 text-white p-4 hover:bg-blue-600"
          >
            /
          </button>
          <button
            onClick={() => handleOperator('*')}
            className="bg-blue-500 text-white p-4 hover:bg-blue-600"
          >
            ×
          </button>
          <button
            onClick={() => handleOperator('-')}
            className="bg-blue-500 text-white p-4 hover:bg-blue-600"
          >
            −
          </button>
          <button
            onClick={() => handleNumber('7')}
            className="bg-gray-200 p-4 hover:bg-gray-300"
          >
            7
          </button>
          <button
            onClick={() => handleNumber('8')}
            className="bg-gray-200 p-4 hover:bg-gray-300"
          >
            8
          </button>
          <button
            onClick={() => handleNumber('9')}
            className="bg-gray-200 p-4 hover:bg-gray-300"
          >
            9
          </button>
          <button
            onClick={() => handleOperator('+')}
            className="bg-blue-500 text-white p-4 hover:bg-blue-600"
          >
            +
          </button>
          <button
            onClick={() => handleNumber('4')}
            className="bg-gray-200 p-4 hover:bg-gray-300"
          >
            4
          </button>
          <button
            onClick={() => handleNumber('5')}
            className="bg-gray-200 p-4 hover:bg-gray-300"
          >
            5
          </button>
          <button
            onClick={() => handleNumber('6')}
            className="bg-gray-200 p-4 hover:bg-gray-300"
          >
            6
          </button>
          <button
            onClick={() => handleOperator('+')}
            className="bg-blue-500 text-white p-4 hover:bg-blue-600"
          >
            +
          </button>
          <button
            onClick={() => handleNumber('1')}
            className="bg-gray-200 p-4 hover:bg-gray-300"
          >
            1
          </button>
          <button
            onClick={() => handleNumber('2')}
            className="bg-gray-200 p-4 hover:bg-gray-300"
          >
            2
          </button>
          <button
            onClick={() => handleNumber('3')}
            className="bg-gray-200 p-4 hover:bg-gray-300"
          >
            3
          </button>
          <button
            onClick={() => handleOperator('-')}
            className="bg-blue-500 text-white p-4 hover:bg-blue-600"
          >
            −
          </button>
          <button
            onClick={() => handleNumber('0')}
            className="col-span-2 bg-gray-200 p-4 hover:bg-gray-300"
          >
            0
          </button>
          <button
            onClick={handleDecimal}
            className="bg-gray-200 p-4 hover:bg-gray-300"
          >
            .
          </button>
          <button
            onClick={handleEquals}
            className="bg-blue-500 text-white p-4 hover:bg-blue-600"
          >
            =
          </button>
        </div>
      </div>
    </div>
  );
}