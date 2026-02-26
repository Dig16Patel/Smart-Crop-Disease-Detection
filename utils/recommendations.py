recommendations = {
    "Tomato_Early_Blight": {
        "description": "Early blight involves concentric rings on lower leaves, turning yellow and dropping off.",
        "treatment": [
            "Use Mancozeb or Chlorothalonil fungicide.",
            "Improve air circulation between plants.",
            "Remove and destroy infected leaves immediately.",
            "Water at the base of the plant, not overhead."
        ],
        "severity": "Moderate"
    },
    "Tomato_Late_Blight": {
        "description": "Late blight causes dark, water-soaked spots on leaves and white fungal growth on undersides.",
        "treatment": [
            "Apply copper-based fungicides.",
            "Destroy all infected plants immediately to prevent spread.",
            "Keep foliage dry; avoid overhead irrigation."
        ],
        "severity": "High"
    },
    "Tomato_Healthy": {
        "description": "The plant appears healthy with no visible signs of disease.",
        "treatment": [
            "Continue regular watering and fertilization.",
            "Monitor regularly for any signs of pests or diseases."
        ],
        "severity": "None"
    },
     "Potato_Early_Blight": {
        "description": "Brown spots with concentric rings on older leaves.",
        "treatment": [
            "Apply fungicides like Mancozeb.",
            "Practice crop rotation.",
            "Ensure proper nitrogen fertilization."
        ],
        "severity": "Moderate"
    },
    "Potato_Late_Blight": {
        "description": "Rapidly spreading dark lesions on leaves and stems.",
        "treatment": [
            "Apply fungicides immediately.",
            "Destroy infected tubers.",
            "Ensure good drainage."
        ],
        "severity": "High"
    },
    "Potato_Healthy": {
        "description": "No disease detected.",
        "treatment": [
            "Maintain good agricultural practices."
        ],
        "severity": "None"
    }
}

def get_recommendation(disease_name):
    """
    Returns the recommendation info for a given disease.
    """
    return recommendations.get(disease_name, {
        "description": "Disease not recognized.",
        "treatment": ["Consult an expert agriculturalist."],
        "severity": "Unknown"
    })

if __name__ == "__main__":
    # Test the module
    disease = "Tomato_Early_Blight"
    info = get_recommendation(disease)
    print(f"Disease: {disease}")
    print(f"Severity: {info['severity']}")
    print(f"Treatment: {info['treatment']}")
