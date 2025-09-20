from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import httpx
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import bcrypt
import jwt
from passlib.context import CryptContext


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.environ.get("JWT_SECRET", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Weather API Key (you'll need to get this from OpenWeatherMap)
WEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "your-openweather-api-key")

# Define Models
class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    name: str
    phone: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserCreate(BaseModel):
    email: str
    password: str
    name: str
    phone: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class FarmDetails(BaseModel):
    state: str
    district: str
    village: str
    pincode: str
    farm_size: float
    farm_size_unit: str  # acre/bigha/hectare

class CropInfo(BaseModel):
    crop_name: str
    variety: str
    sowing_date: str
    season: str  # Kharif/Rabi/Zaid

class SoilInputs(BaseModel):
    soil_type: str
    fertilizer_used: str
    ph_level: Optional[float] = None
    organic_carbon: Optional[float] = None

class IrrigationInfo(BaseModel):
    irrigation_source: str
    irrigation_frequency: str
    water_availability: str

class CropPredictionRequest(BaseModel):
    user_id: str
    farm_details: FarmDetails
    crop_info: CropInfo
    soil_inputs: SoilInputs
    irrigation_info: IrrigationInfo

class CropPredictionResponse(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    predicted_yield: float
    yield_unit: str
    district_average: float
    comparison_percentage: float
    recommendations: List[str]
    confidence_score: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class WeatherData(BaseModel):
    location: str
    temperature: float
    humidity: float
    rainfall: float
    description: str

class ChatMessage(BaseModel):
    message: str
    language: str = "en"  # en, hi, bn, or (odia)

class ChatResponse(BaseModel):
    response: str
    language: str
    recommendations: Optional[List[str]] = None


# Authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_email: str = payload.get("sub")
        if user_email is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        user = await db.users.find_one({"email": user_email})
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return User(**user)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")


# Sample agricultural data for prediction model
def get_sample_agricultural_data():
    """Generate sample agricultural data for training the ML model"""
    np.random.seed(42)
    n_samples = 1000
    
    # Features: farm_size, rainfall, temperature, humidity, soil_ph, fertilizer_amount
    data = {
        'farm_size': np.random.uniform(0.5, 10, n_samples),
        'rainfall': np.random.uniform(400, 1200, n_samples),
        'temperature': np.random.uniform(20, 35, n_samples),
        'humidity': np.random.uniform(40, 90, n_samples),
        'soil_ph': np.random.uniform(5.5, 8.5, n_samples),
        'fertilizer_amount': np.random.uniform(50, 200, n_samples),
        'irrigation_frequency': np.random.randint(1, 4, n_samples),  # 1=Rarely, 2=Sometimes, 3=Regularly
    }
    
    # Target: yield (quintals per hectare)
    # Simple formula to generate realistic yields
    yield_base = (
        data['farm_size'] * 0.5 +
        data['rainfall'] * 0.01 +
        (35 - abs(data['temperature'] - 27)) * 0.3 +
        data['humidity'] * 0.05 +
        (7 - abs(data['soil_ph'] - 7)) * 2 +
        data['fertilizer_amount'] * 0.02 +
        data['irrigation_frequency'] * 2
    )
    
    # Add some noise
    data['yield'] = yield_base + np.random.normal(0, 2, n_samples)
    data['yield'] = np.maximum(data['yield'], 5)  # Minimum yield of 5 quintals
    
    return pd.DataFrame(data)

# Initialize and train the ML model
df = get_sample_agricultural_data()
X = df.drop('yield', axis=1)
y = df['yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# State and district data for India
INDIAN_STATES_DISTRICTS = {
    "Andhra Pradesh": ["Anantapur", "Chittoor", "East Godavari", "Guntur", "Krishna", "Kurnool", "Prakasam", "Srikakulam", "Visakhapatnam", "Vizianagaram", "West Godavari", "YSR Kadapa"],
    "Arunachal Pradesh": ["Anjaw", "Changlang", "Dibang Valley", "East Kameng", "East Siang", "Kurung Kumey", "Lohit", "Lower Dibang Valley", "Lower Subansiri", "Papum Pare", "Tawang", "Tirap", "Upper Siang", "Upper Subansiri", "West Kameng", "West Siang"],
    "Assam": ["Baksa", "Barpeta", "Biswanath", "Bongaigaon", "Cachar", "Charaideo", "Chirang", "Darrang", "Dhemaji", "Dhubri", "Dibrugarh", "Goalpara", "Golaghat", "Hailakandi", "Hojai", "Jorhat", "Kamrup", "Kamrup Metropolitan", "Karbi Anglong", "Karimganj", "Kokrajhar", "Lakhimpur", "Majuli", "Morigaon", "Nagaon", "Nalbari", "Dima Hasao", "Sivasagar", "Sonitpur", "South Salmara-Mankachar", "Tinsukia", "Udalguri", "West Karbi Anglong"],
    "Bihar": ["Araria", "Arwal", "Aurangabad", "Banka", "Begusarai", "Bhagalpur", "Bhojpur", "Buxar", "Darbhanga", "East Champaran", "Gaya", "Gopalganj", "Jamui", "Jehanabad", "Kaimur", "Katihar", "Khagaria", "Kishanganj", "Lakhisarai", "Madhepura", "Madhubani", "Munger", "Muzaffarpur", "Nalanda", "Nawada", "Patna", "Purnia", "Rohtas", "Saharsa", "Samastipur", "Saran", "Sheikhpura", "Sheohar", "Sitamarhi", "Siwan", "Supaul", "Vaishali", "West Champaran"],
    "Chhattisgarh": ["Balod", "Baloda Bazar", "Balrampur", "Bastar", "Bemetara", "Bijapur", "Bilaspur", "Dantewada", "Dhamtari", "Durg", "Gariaband", "Janjgir-Champa", "Jashpur", "Kabirdham", "Kanker", "Kondagaon", "Korba", "Korea", "Mahasamund", "Mungeli", "Narayanpur", "Raigarh", "Raipur", "Rajnandgaon", "Sukma", "Surajpur", "Surguja"],
    "Goa": ["North Goa", "South Goa"],
    "Gujarat": ["Ahmedabad", "Amreli", "Anand", "Aravalli", "Banaskantha", "Bharuch", "Bhavnagar", "Botad", "Chhota Udepur", "Dahod", "Dang", "Gandhinagar", "Gir Somnath", "Jamnagar", "Junagadh", "Kutch", "Kheda", "Mahisagar", "Mehsana", "Morbi", "Narmada", "Navsari", "Panchmahal", "Patan", "Porbandar", "Rajkot", "Sabarkantha", "Surat", "Surendranagar", "Tapi", "Vadodara", "Valsad"],
    "Haryana": ["Ambala", "Bhiwani", "Charkhi Dadri", "Faridabad", "Fatehabad", "Gurgaon", "Hisar", "Jhajjar", "Jind", "Kaithal", "Karnal", "Kurukshetra", "Mahendragarh", "Mewat", "Palwal", "Panchkula", "Panipat", "Rewari", "Rohtak", "Sirsa", "Sonipat", "Yamunanagar"],
    "Himachal Pradesh": ["Bilaspur", "Chamba", "Hamirpur", "Kangra", "Kinnaur", "Kullu", "Lahaul and Spiti", "Mandi", "Shimla", "Sirmaur", "Solan", "Una"],
    "Jharkhand": ["Bokaro", "Chatra", "Deoghar", "Dhanbad", "Dumka", "East Singhbhum", "Garhwa", "Giridih", "Godda", "Gumla", "Hazaribagh", "Jamtara", "Khunti", "Koderma", "Latehar", "Lohardaga", "Pakur", "Palamu", "Ramgarh", "Ranchi", "Sahibganj", "Seraikela Kharsawan", "Simdega", "West Singhbhum"],
    "Karnataka": ["Bagalkot", "Ballari", "Belagavi", "Bengaluru Rural", "Bengaluru Urban", "Bidar", "Chamarajanagar", "Chikballapur", "Chikkamagaluru", "Chitradurga", "Dakshina Kannada", "Davanagere", "Dharwad", "Gadag", "Hassan", "Haveri", "Kalaburagi", "Kodagu", "Kolar", "Koppal", "Mandya", "Mysuru", "Raichur", "Ramanagara", "Shivamogga", "Tumakuru", "Udupi", "Uttara Kannada", "Vijayapura", "Yadgir"],
    "Kerala": ["Alappuzha", "Ernakulam", "Idukki", "Kannur", "Kasaragod", "Kollam", "Kottayam", "Kozhikode", "Malappuram", "Palakkad", "Pathanamthitta", "Thiruvananthapuram", "Thrissur", "Wayanad"],
    "Madhya Pradesh": ["Agar Malwa", "Alirajpur", "Anuppur", "Ashoknagar", "Balaghat", "Barwani", "Betul", "Bhind", "Bhopal", "Burhanpur", "Chhatarpur", "Chhindwara", "Damoh", "Datia", "Dewas", "Dhar", "Dindori", "Guna", "Gwalior", "Harda", "Hoshangabad", "Indore", "Jabalpur", "Jhabua", "Katni", "Khandwa", "Khargone", "Mandla", "Mandsaur", "Morena", "Narsinghpur", "Neemuch", "Panna", "Raisen", "Rajgarh", "Ratlam", "Rewa", "Sagar", "Satna", "Sehore", "Seoni", "Shahdol", "Shajapur", "Sheopur", "Shivpuri", "Sidhi", "Singrauli", "Tikamgarh", "Ujjain", "Umaria", "Vidisha"],
    "Maharashtra": ["Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed", "Bhandara", "Buldhana", "Chandrapur", "Dhule", "Gadchiroli", "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur", "Latur", "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded", "Nandurbar", "Nashik", "Osmanabad", "Palghar", "Parbhani", "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara", "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal"],
    "Manipur": ["Bishnupur", "Chandel", "Churachandpur", "Imphal East", "Imphal West", "Jiribam", "Kakching", "Kamjong", "Kangpokpi", "Noney", "Pherzawl", "Senapati", "Tamenglong", "Tengnoupal", "Thoubal", "Ukhrul"],
    "Meghalaya": ["East Garo Hills", "East Jaintia Hills", "East Khasi Hills", "North Garo Hills", "Ri Bhoi", "South Garo Hills", "South West Garo Hills", "South West Khasi Hills", "West Garo Hills", "West Jaintia Hills", "West Khasi Hills"],
    "Mizoram": ["Aizawl", "Champhai", "Kolasib", "Lawngtlai", "Lunglei", "Mamit", "Saiha", "Serchhip"],
    "Nagaland": ["Dimapur", "Kiphire", "Kohima", "Longleng", "Mokokchung", "Mon", "Peren", "Phek", "Tuensang", "Wokha", "Zunheboto"],
    "Odisha": ["Angul", "Balangir", "Balasore", "Bargarh", "Bhadrak", "Boudh", "Cuttack", "Deogarh", "Dhenkanal", "Gajapati", "Ganjam", "Jagatsinghpur", "Jajpur", "Jharsuguda", "Kalahandi", "Kandhamal", "Kendrapara", "Kendujhar", "Khordha", "Koraput", "Malkangiri", "Mayurbhanj", "Nabarangpur", "Nayagarh", "Nuapada", "Puri", "Rayagada", "Sambalpur", "Sonepur", "Sundargarh"],
    "Punjab": ["Amritsar", "Barnala", "Bathinda", "Faridkot", "Fatehgarh Sahib", "Fazilka", "Ferozepur", "Gurdaspur", "Hoshiarpur", "Jalandhar", "Kapurthala", "Ludhiana", "Mansa", "Moga", "Muktsar", "Nawanshahr", "Pathankot", "Patiala", "Rupnagar", "Sahibzada Ajit Singh Nagar", "Sangrur", "Tarn Taran"],
    "Rajasthan": ["Ajmer", "Alwar", "Banswara", "Baran", "Barmer", "Bharatpur", "Bhilwara", "Bikaner", "Bundi", "Chittorgarh", "Churu", "Dausa", "Dholpur", "Dungarpur", "Hanumangarh", "Jaipur", "Jaisalmer", "Jalore", "Jhalawar", "Jhunjhunu", "Jodhpur", "Karauli", "Kota", "Nagaur", "Pali", "Pratapgarh", "Rajsamand", "Sawai Madhopur", "Sikar", "Sirohi", "Sri Ganganagar", "Tonk", "Udaipur"],
    "Sikkim": ["East Sikkim", "North Sikkim", "South Sikkim", "West Sikkim"],
    "Tamil Nadu": ["Ariyalur", "Chennai", "Coimbatore", "Cuddalore", "Dharmapuri", "Dindigul", "Erode", "Kanchipuram", "Kanyakumari", "Karur", "Krishnagiri", "Madurai", "Nagapattinam", "Namakkal", "Nilgiris", "Perambalur", "Pudukkottai", "Ramanathapuram", "Salem", "Sivaganga", "Thanjavur", "Theni", "Thoothukudi", "Tiruchirappalli", "Tirunelveli", "Tiruppur", "Tiruvallur", "Tiruvannamalai", "Tiruvarur", "Vellore", "Viluppuram", "Virudhunagar"],
    "Telangana": ["Adilabad", "Bhadradri Kothagudem", "Hyderabad", "Jagtial", "Jangaon", "Jayashankar Bhupalpally", "Jogulamba Gadwal", "Kamareddy", "Karimnagar", "Khammam", "Komaram Bheem Asifabad", "Mahabubabad", "Mahabubnagar", "Mancherial", "Medak", "Medchal", "Nagarkurnool", "Nalgonda", "Nirmal", "Nizamabad", "Peddapalli", "Rajanna Sircilla", "Rangareddy", "Sangareddy", "Siddipet", "Suryapet", "Vikarabad", "Wanaparthy", "Warangal Rural", "Warangal Urban", "Yadadri Bhuvanagiri"],
    "Tripura": ["Dhalai", "Gomati", "Khowai", "North Tripura", "Sepahijala", "South Tripura", "Unakoti", "West Tripura"],
    "Uttar Pradesh": ["Agra", "Aligarh", "Allahabad", "Ambedkar Nagar", "Amethi", "Amroha", "Auraiya", "Azamgarh", "Baghpat", "Bahraich", "Ballia", "Balrampur", "Banda", "Barabanki", "Bareilly", "Basti", "Bhadohi", "Bijnor", "Budaun", "Bulandshahr", "Chandauli", "Chitrakoot", "Deoria", "Etah", "Etawah", "Faizabad", "Farrukhabad", "Fatehpur", "Firozabad", "Gautam Buddha Nagar", "Ghaziabad", "Ghazipur", "Gonda", "Gorakhpur", "Hamirpur", "Hapur", "Hardoi", "Hathras", "Jalaun", "Jaunpur", "Jhansi", "Kannauj", "Kanpur Dehat", "Kanpur Nagar", "Kasganj", "Kaushambi", "Kheri", "Kushinagar", "Lalitpur", "Lucknow", "Maharajganj", "Mahoba", "Mainpuri", "Mathura", "Mau", "Meerut", "Mirzapur", "Moradabad", "Muzaffarnagar", "Pilibhit", "Pratapgarh", "Raebareli", "Rampur", "Saharanpur", "Sambhal", "Sant Kabir Nagar", "Shahjahanpur", "Shamli", "Shravasti", "Siddharthnagar", "Sitapur", "Sonbhadra", "Sultanpur", "Unnao", "Varanasi"],
    "Uttarakhand": ["Almora", "Bageshwar", "Chamoli", "Champawat", "Dehradun", "Haridwar", "Nainital", "Pauri Garhwal", "Pithoragarh", "Rudraprayag", "Tehri Garhwal", "Udham Singh Nagar", "Uttarkashi"],
    "West Bengal": ["Alipurduar", "Bankura", "Birbhum", "Cooch Behar", "Dakshin Dinajpur", "Darjeeling", "Hooghly", "Howrah", "Jalpaiguri", "Jhargram", "Kalimpong", "Kolkata", "Malda", "Murshidabad", "Nadia", "North 24 Parganas", "Paschim Bardhaman", "Paschim Medinipur", "Purba Bardhaman", "Purba Medinipur", "Purulia", "South 24 Parganas", "Uttar Dinajpur"]
}

CROP_TYPES = [
    "Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Soybean", "Groundnut", 
    "Sunflower", "Mustard", "Chickpea", "Pigeon Pea", "Black Gram", "Green Gram",
    "Pearl Millet", "Sorghum", "Finger Millet", "Barley", "Sesame", "Castor",
    "Onion", "Potato", "Tomato", "Brinjal", "Okra", "Chilli"
]

SOIL_TYPES = [
    {"name": "Sandy", "description": "Well-drained, easy to cultivate"},
    {"name": "Loamy", "description": "Best for most crops, balanced nutrients"},
    {"name": "Clay", "description": "High water retention, nutrient-rich"},
    {"name": "Black Cotton", "description": "High fertility, good for cotton and wheat"},
    {"name": "Red", "description": "Good drainage, suitable for millets"},
    {"name": "Alluvial", "description": "Very fertile, good for rice and wheat"}
]

# Translation dictionaries
TRANSLATIONS = {
    'en': {
        'welcome': 'Welcome to HarvestGuru',
        'prediction_result': 'Crop Yield Prediction Result',
        'recommendations': 'Recommendations',
        'weather_advice': 'Weather-based Advice',
    },
    'hi': {
        'welcome': 'हार्वेस्टगुरु में आपका स्वागत है',
        'prediction_result': 'फसल उपज पूर्वानुमान परिणाम',
        'recommendations': 'सिफारिशें',
        'weather_advice': 'मौसम आधारित सलाह',
    },
    'bn': {
        'welcome': 'হার্ভেস্টগুরুতে স্বাগতম',
        'prediction_result': 'ফসলের ফলন পূর্বাভাসের ফলাফল',
        'recommendations': 'সুপারিশ',
        'weather_advice': 'আবহাওয়া ভিত্তিক পরামর্শ',
    },
    'or': {  # Odia
        'welcome': 'ହାର্ଭେଷ୍ଟଗୁରୁକୁ ସ୍ୱାଗତ',
        'prediction_result': 'ଫସଲ ଉତ୍ପାଦନ ପୂର୍ବାନୁମାନ ଫଳାଫଳ',
        'recommendations': 'ସୁପାରିଶ',
        'weather_advice': 'ପାଣିପାଗ ଆଧାରିତ ପରାମର୍ଶ',
    }
}


# Helper functions
async def get_weather_data(latitude: float, longitude: float):
    """Get weather data from OpenWeatherMap API"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://api.openweathermap.org/data/2.5/weather",
                params={
                    "lat": latitude,
                    "lon": longitude,
                    "appid": WEATHER_API_KEY,
                    "units": "metric"
                },
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "temperature": data["main"]["temp"],
                    "humidity": data["main"]["humidity"],
                    "description": data["weather"][0]["description"],
                    "rainfall": data.get("rain", {}).get("1h", 0)
                }
    except Exception as e:
        # Return mock weather data if API fails
        return {
            "temperature": 28.5,
            "humidity": 65,
            "description": "partly cloudy",
            "rainfall": 2.5
        }

def predict_crop_yield(farm_details, crop_info, soil_inputs, irrigation_info, weather_data):
    """Predict crop yield using the trained ML model"""
    
    # Map categorical data to numerical values
    irrigation_freq_map = {"Rarely": 1, "Sometimes": 2, "Regularly": 3}
    
    # Prepare features for prediction
    features = np.array([[
        farm_details.farm_size,  # Convert to hectares if needed
        weather_data.get("rainfall", 50) * 10,  # Simulate annual rainfall
        weather_data.get("temperature", 28),
        weather_data.get("humidity", 65),
        soil_inputs.ph_level or 7.0,  # Default pH
        100,  # Default fertilizer amount
        irrigation_freq_map.get(irrigation_info.irrigation_frequency, 2)
    ]])
    
    predicted_yield = model.predict(features)[0]
    confidence = model.score(X_test, y_test)
    
    return predicted_yield, confidence

def generate_recommendations(crop_info, soil_inputs, irrigation_info, weather_data, predicted_yield, district_avg=15):
    """Generate farming recommendations based on input data"""
    recommendations = []
    
    # Weather-based recommendations
    if weather_data.get("temperature", 28) > 32:
        recommendations.append("High temperature detected. Increase irrigation frequency during peak hours.")
    
    if weather_data.get("humidity", 65) < 40:
        recommendations.append("Low humidity levels. Consider mulching to retain soil moisture.")
    
    # Soil-based recommendations
    if soil_inputs.ph_level and soil_inputs.ph_level < 6.0:
        recommendations.append("Soil is acidic. Apply lime to improve pH levels.")
    elif soil_inputs.ph_level and soil_inputs.ph_level > 8.0:
        recommendations.append("Soil is alkaline. Apply organic matter to balance pH.")
    
    # Irrigation recommendations
    if irrigation_info.irrigation_frequency == "Rarely":
        recommendations.append("Consider increasing irrigation frequency for better yield.")
    
    # Fertilizer recommendations
    if soil_inputs.fertilizer_used == "None":
        recommendations.append("Apply balanced NPK fertilizer for optimal growth.")
    
    # Yield-based recommendations
    if predicted_yield < district_avg * 0.8:
        recommendations.append("Predicted yield is below district average. Consider soil testing and improved seeds.")
    
    return recommendations


# Authentication routes
@api_router.post("/auth/register", response_model=Token)
async def register(user_data: UserCreate):
    # Check if user already exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password and create user
    hashed_password = get_password_hash(user_data.password)
    user = User(
        email=user_data.email,
        name=user_data.name,
        phone=user_data.phone
    )
    
    user_dict = user.dict()
    user_dict["password_hash"] = hashed_password
    
    await db.users.insert_one(user_dict)
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@api_router.post("/auth/login", response_model=Token)
async def login(user_credentials: UserLogin):
    # Find user
    user = await db.users.find_one({"email": user_credentials.email})
    if not user or not verify_password(user_credentials.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@api_router.get("/auth/me", response_model=User)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user


# Main application routes
@api_router.get("/")
async def root():
    return {"message": "HarvestGuru API - AI-powered Crop Yield Prediction"}

@api_router.get("/states")
async def get_states():
    return {"states": list(INDIAN_STATES_DISTRICTS.keys())}

@api_router.get("/districts/{state}")
async def get_districts(state: str):
    districts = INDIAN_STATES_DISTRICTS.get(state, [])
    return {"districts": districts}

@api_router.get("/crops")
async def get_crops():
    return {"crops": CROP_TYPES}

@api_router.get("/soil-types")
async def get_soil_types():
    return {"soilTypes": SOIL_TYPES}

@api_router.get("/weather/{latitude}/{longitude}")
async def get_weather(latitude: float, longitude: float):
    weather_data = await get_weather_data(latitude, longitude)
    return weather_data

@api_router.post("/predict-yield", response_model=CropPredictionResponse)
async def predict_yield(
    prediction_request: CropPredictionRequest,
    current_user: User = Depends(get_current_user)
):
    # Get weather data (mock coordinates for demo)
    weather_data = await get_weather_data(20.5937, 78.9629)  # Center of India
    
    # Predict yield
    predicted_yield, confidence = predict_crop_yield(
        prediction_request.farm_details,
        prediction_request.crop_info,
        prediction_request.soil_inputs,
        prediction_request.irrigation_info,
        weather_data
    )
    
    # Get district average (mock data)
    district_avg = np.random.uniform(12, 18)  # Mock district average
    comparison = ((predicted_yield - district_avg) / district_avg) * 100
    
    # Generate recommendations
    recommendations = generate_recommendations(
        prediction_request.crop_info,
        prediction_request.soil_inputs,
        prediction_request.irrigation_info,
        weather_data,
        predicted_yield,
        district_avg
    )
    
    # Create response
    response = CropPredictionResponse(
        predicted_yield=round(predicted_yield, 2),
        yield_unit="quintals per hectare",
        district_average=round(district_avg, 2),
        comparison_percentage=round(comparison, 1),
        recommendations=recommendations,
        confidence_score=round(confidence * 100, 1)
    )
    
    # Save prediction to database
    prediction_dict = response.dict()
    prediction_dict["user_id"] = current_user.id
    prediction_dict["input_data"] = prediction_request.dict()
    await db.predictions.insert_one(prediction_dict)
    
    return response

@api_router.get("/my-predictions")
async def get_my_predictions(current_user: User = Depends(get_current_user)):
    predictions = await db.predictions.find({"user_id": current_user.id}).to_list(100)
    return {"predictions": predictions}

@api_router.post("/chat", response_model=ChatResponse)
async def chat_with_bot(message: ChatMessage, current_user: User = Depends(get_current_user)):
    """Simple multilingual chatbot for farming advice"""
    
    # Simple keyword-based responses
    user_message = message.message.lower()
    language = message.language
    
    # Default responses in different languages
    responses = {
        'en': {
            'greeting': "Hello! I'm your farming assistant. How can I help you today?",
            'weather': "For weather updates, please check the weather section in your dashboard.",
            'irrigation': "For optimal irrigation, water your crops early morning or evening. Avoid midday watering.",
            'fertilizer': "Use balanced NPK fertilizers. Apply based on soil test results.",
            'pest': "Regular monitoring and organic pest control methods are recommended.",
            'default': "I can help you with farming advice, weather information, and crop recommendations."
        },
        'hi': {
            'greeting': "नमस्ते! मैं आपका कृषि सहायक हूं। आज मैं आपकी कैसे मदद कर सकता हूं?",
            'weather': "मौसम अपडेट के लिए, कृपया अपने डैशबोर्ड में मौसम अनुभाग देखें।",
            'irrigation': "सिंचाई के लिए सुबह या शाम का समय बेहतर है। दोपहर में सिंचाई से बचें।",
            'fertilizer': "संतुलित NPK उर्वरक का उपयोग करें। मिट्टी परीक्षण के आधार पर लगाएं।",
            'pest': "नियमित निगरानी और जैविक कीट नियंत्रण विधियों की सिफारिश की जाती है।",
            'default': "मैं आपको कृषि सलाह, मौसम की जानकारी और फसल सिफारिशों में मदद कर सकता हूं।"
        },
        'bn': {
            'greeting': "নমস্কার! আমি আপনার কৃষি সহায়ক। আজ আমি কীভাবে আপনাকে সাহায্য করতে পারি?",
            'weather': "আবহাওয়ার আপডেটের জন্য, দয়া করে আপনার ড্যাশবোর্ডের আবহাওয়া বিভাগটি দেখুন।",
            'irrigation': "সেচের জন্য ভোর বা সন্ধ্যার সময় ভাল। দুপুরে সেচ এড়িয়ে চলুন।",
            'fertilizer': "সুষম NPK সার ব্যবহার করুন। মাটি পরীক্ষার ভিত্তিতে প্রয়োগ করুন।",
            'pest': "নিয়মিত পর্যবেক্ষণ এবং জৈব কীটপতঙ্গ নিয়ন্ত্রণ পদ্ধতি সুপারিশ করা হয়।",
            'default': "আমি আপনাকে কৃষি পরামর্শ, আবহাওয়ার তথ্য এবং ফসলের সুপারিশে সাহায্য করতে পারি।"
        },
        'or': {  # Odia
            'greeting': "ନମସ୍କାର! ମୁଁ ଆପଣଙ୍କର କୃଷି ସହାୟକ। ଆଜି ମୁଁ କିପରି ଆପଣଙ୍କୁ ସାହାଯ୍ୟ କରିପାରିବି?",
            'weather': "ପାଣିପାଗ ଅପଡେଟ୍ ପାଇଁ, ଦୟାକରି ଆପଣଙ୍କ ଡ୍ୟାସବୋର୍ଡରେ ପାଣିପାଗ ବିଭାଗ ଦେଖନ୍ତୁ।",
            'irrigation': "ଜଳସେଚନ ପାଇଁ ସକାଳ କିମ୍ବା ସନ୍ଧ୍ୟା ସମୟ ଭଲ। ମଧ୍ୟାହ୍ନରେ ଜଳସେଚନ ଏଡାନ୍ତୁ।",
            'fertilizer': "ସନ୍ତୁଳିତ NPK ସାର ବ୍ୟବହାର କରନ୍ତୁ। ମାଟି ପରୀକ୍ଷା ଆଧାରରେ ପ୍ରୟୋଗ କରନ୍ତୁ।",
            'pest': "ନିୟମିତ ନଜର ରଖିବା ଏବଂ ଜୈବିକ କୀଟ ନିୟନ୍ତ୍ରଣ ପଦ୍ଧତି ସୁପାରିଶ କରାଯାଏ।",
            'default': "ମୁଁ ଆପଣଙ୍କୁ କୃଷି ପରାମର୍ଶ, ପାଣିପାଗ ସୂଚନା ଏବଂ ଫସଲ ସୁପାରିଶରେ ସାହାଯ୍ୟ କରିପାରିବି।"
        }
    }
    
    # Simple keyword matching
    response_text = ""
    if any(word in user_message for word in ['hello', 'hi', 'namaste', 'namaskar']):
        response_text = responses[language]['greeting']
    elif any(word in user_message for word in ['weather', 'rain', 'temperature']):
        response_text = responses[language]['weather']
    elif any(word in user_message for word in ['irrigation', 'water', 'sech']):
        response_text = responses[language]['irrigation']
    elif any(word in user_message for word in ['fertilizer', 'manure', 'sarr']):
        response_text = responses[language]['fertilizer']
    elif any(word in user_message for word in ['pest', 'insect', 'keet']):
        response_text = responses[language]['pest']
    else:
        response_text = responses[language]['default']
    
    return ChatResponse(
        response=response_text,
        language=language,
        recommendations=["Check your dashboard for personalized recommendations", "Visit weather section for updates"]
    )


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()