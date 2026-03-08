from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum


class ContractTypeEnum(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class InternetServiceEnum(str, Enum):
    dsl = "DSL"
    fiber_optic = "Fiber optic"
    no = "No"


class PaymentMethodEnum(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer (automatic)"
    credit_card = "Credit card (automatic)"


class YesNo(str, Enum):
    yes = "Yes"
    no = "No"


class YesNoNoService(str, Enum):
    yes = "Yes"
    no = "No"
    no_internet = "No internet service"
    no_phone = "No phone service"


class CustomerFeatures(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(..., ge=0, le=120)
    PhoneService: YesNo
    MultipleLines: YesNoNoService
    InternetService: InternetServiceEnum   # ← field name differs from class name
    OnlineSecurity: YesNoNoService
    OnlineBackup: YesNoNoService
    DeviceProtection: YesNoNoService
    TechSupport: YesNoNoService
    StreamingTV: YesNoNoService
    StreamingMovies: YesNoNoService
    Contract: ContractTypeEnum
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethodEnum       # ← field name differs from class name
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    class Config:
        use_enum_values = True


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskFactor(BaseModel):
    feature: str
    importance: float
    value: str


class PredictionResponse(BaseModel):
    churn: bool
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    top_risk_factors: list[RiskFactor]


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1)


class BatchPrediction(BaseModel):
    index: int
    churn: bool
    churn_probability: float
    risk_level: RiskLevel


class BatchResponse(BaseModel):
    total: int
    churn_count: int
    churn_rate: float
    predictions: list[BatchPrediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]