import React from 'react'
import { useState } from 'react';
import './home.css'

const HomePage = () => {
    const [features, setFeatures] = useState({
        fixedAcidity: '',
        volatileAcidity: '',
        citricAcid: '',
        residualSugar: '',
        chlorides: '',
        freeSulfurDioxide: '',
        totalSulfurDioxide: '',
        density: '',
        pH: '',
        sulphates: '',
        alcohol: ''
      });
      const [result, setResult] = useState('');
    
      // Handle changes in input fields
      const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFeatures((prev) => ({
          ...prev,
          [name]: value
        }));
      };
    
      // Handle form submission
      const handleSubmit = async (e) => {
        e.preventDefault();

        const hasEmptyFields = Object.values(features).some((value)=>value == '');
        
        if (hasEmptyFields) {
          setResult('Error: Please fill in all fields');
          return;
        }
        
        try {
          // Send the features to the Flask backend via a POST request
          const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ features: Object.values(features) })
          });
          const data = await response.json();
          
          // Set the result (High/Low) based on the response
          if (data.quality) {
            setResult(`Wine quality is: ${data.quality}`);
            setFeatures(
                {
                  fixedAcidity: '',
                  volatileAcidity: '',
                  citricAcid: '',
                  residualSugar: '',
                  chlorides: '',
                  freeSulfurDioxide: '',
                  totalSulfurDioxide: '',
                  density: '',
                  pH: '',
                  sulphates: '',
                  alcohol: ''
                }
              
            )
          }else {
            setResult('Error: Unable to predict the wine quality');
          }
        } catch (error) {
          setResult('Error: Failed to fetch data from server');
          console.error(error);
        }
      };
    
  return (
    <div>
      <h1>Test Your Wine, Discover Its Quality Instantly!</h1>

      <form onSubmit={handleSubmit}>
        <div>
          <label>Fixed Acidity:</label>
          <input
            type="number"
            name="fixedAcidity"
            value={features.fixedAcidity}
            onChange={handleInputChange}
          />
        </div>
        <div>
          <label>Volatile Acidity:</label>
          <input
            type="number"
            name="volatileAcidity"
            value={features.volatileAcidity}
            onChange={handleInputChange}
          />
        </div>
        <div>
          <label>Citric Acid:</label>
          <input
            type="number"
            name="citricAcid"
            value={features.citricAcid}
            onChange={handleInputChange}
          />
        </div>
        <div>
          <label>Residual Sugar:</label>
          <input
            type="number"
            name="residualSugar"
            value={features.residualSugar}
            onChange={handleInputChange}
          />
        </div>
        <div>
          <label>Chlorides:</label>
          <input
            type="number"
            name="chlorides"
            value={features.chlorides}
            onChange={handleInputChange}
          />
        </div>
        <div>
          <label>Free Sulfur Dioxide:</label>
          <input
            type="number"
            name="freeSulfurDioxide"
            value={features.freeSulfurDioxide}
            onChange={handleInputChange}
          />
        </div>
        <div>
          <label>Total Sulfur Dioxide:</label>
          <input
            type="number"
            name="totalSulfurDioxide"
            value={features.totalSulfurDioxide}
            onChange={handleInputChange}
          />
        </div>
        <div>
          <label>Density:</label>
          <input
            type="number"
            name="density"
            value={features.density}
            onChange={handleInputChange}
          />
        </div>
        <div>
          <label>pH:</label>
          <input
            type="number"
            name="pH"
            value={features.pH}
            onChange={handleInputChange}
          />
        </div>
        <div>
          <label>Sulphates:</label>
          <input
            type="number"
            name="sulphates"
            value={features.sulphates}
            onChange={handleInputChange}
          />
        </div>
        <div>
          <label>Alcohol:</label>
          <input
            type="number"
            name="alcohol"
            value={features.alcohol}
            onChange={handleInputChange}
          />
        </div>
        <button type="submit">Predict Wine Quality</button>
      </form>
      
      {/* Display the result */}
      {result && <h2>{result}</h2>}
    </div>
  )
}

export default HomePage
