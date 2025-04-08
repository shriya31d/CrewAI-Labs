#!/usr/bin/env python
from datetime import datetime
import json
from random import randint
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from crewai.flow import Flow, listen, start, router, and_
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from azure.identity import DefaultAzureCredential
import mlflow

mlflow.crewai.autolog()
# Optional: Set a tracking URI and an experiment name if you have a tracking server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("CrewAI Flow")

class InvoiceItem(BaseModel):
    name: Optional[str]
    quantity: Optional[int]
    unit_price: Optional[float]

class Invoice(BaseModel):
    customer_name: Optional[str] = Field(default="customer", description="Name of customer")
    items: Optional[List[InvoiceItem]]  = Field(default_factory=list, description="items ordered")
    total_amount: Optional[float] = Field(default=0, description="total amount")
    percentage_discount :Optional[int] = Field(default=0, description="discount")
    date: Optional[str] = Field(default="dd-mm-yyyy", description="date of purchase")
    shipping_charges: Optional[float] = Field(default=100, description="shipping charges")

class OrderState(BaseModel):
    customer_name: Optional[str] = Field(None, description="Name of the customer")
    delivery_location: Optional[str] = Field(default="Pune", description="delivery location")
    order_text: Optional[str] = Field(None, description="Requirement of the customer in natural language")
    requested_items: Optional[Dict[str, int]] = Field(default_factory=dict, description="Items and their quantities")
    unavailable_items: Optional[Dict[str, int]] = Field(default_factory=dict, description="Items not available in stock")
    email_draft: Optional[str] = Field(default="", description="Email to customer regarding stock issues")
    status: Optional[str] = Field(default="placed", description="Status of the order")
    invoice : Optional[Invoice] = Field(default_factory=Invoice, description="Invoice of the order")
    # shipping_charges: Optional[float] = Field(..., description="shipping charges for the order")
    
class SalesFlow(Flow[OrderState]):

    credential = DefaultAzureCredential()
    azure_llm = AzureChatOpenAI(
            api_version= "2023-05-15",
            azure_endpoint="https://<replace>azure.com/",
            deployment_name="gpt-4o",
            azure_ad_token = credential.get_token("https://cognitiveservices.azure.com/.default").token
        )
    with open("<replace>", "r") as f:
        stock_data = json.load(f)

    with open("<replace>", "r") as f:
        prices_data = json.load(f)

    product_names = list(stock_data.keys())

    @start()
    def interpret_order(self):

        prompt = f"""
        You are sales assistant that extracts structured order information from natural language.
        # Available products:
        {json.dumps(self.product_names, indent=2)}

        Your task:
        Map the customer's request to the most relevant products from the list above. Return a json with mapped product names as keys and number of units requested by user.
        
        Please extract the following fields in JSON format:
        - customer_name: str
        - requested_items: dictionary where keys are product names and values are quantities

        Example:
        Input: "I would like to buy 15 iPhones and 3 MacBook Airs."
        Output:
        {{
            "requested_items": {{
                "iPhone 15": 15,
                "MacBook Air": 3
            }}
        }}

        Now process this order:
        "{self.state.order_text}"
        """
        structured_llm = self.azure_llm.with_structured_output(OrderState)
        response = structured_llm.invoke(prompt)
        self.state.requested_items = response.requested_items
        self.state.status = "processing"

    @listen("interpret_order")
    def check_stock_availability(self):
        unavailable = {}

        for product, qty_requested in self.state.requested_items.items():
            stock_qty = self.stock_data.get(product, 0)

            if qty_requested > stock_qty:
                unavailable[product] = qty_requested

            self.state.unavailable_items = unavailable

        if unavailable:
            self.state.status = "stock_issue"
        else:
            self.state.status = "ready_for_dispatch"

        return self.state.status
    
    @router(check_stock_availability)
    def route_based_on_stock(self):
        if self.state.status == "ready_for_dispatch":
            return "success"
        else:
            return "failed"

    # demonstrating parallel execution   
    @listen("success")
    async def send_invoice_on_email(self):
        # You can mock DB save or print here for demonstration
        print(f"EMAIL: Placed order for {self.state.customer_name} with items: {self.state.requested_items}")
        self.state.status = "order_placed"

    @listen("failed")
    def handle_stock_issue(self):
        # You can mock DB save or print here for demonstration
        email_template = f"""
                Dear {self.state.customer_name},

                Thank you for your order. Unfortunately, the following items are currently out of stock:
                
                {', '.join([f"{item} (Qty: {qty})" for item, qty in self.state.unavailable_items.items()])}

                Please let us know if you would like to adjust your order or wait for restocking.

                Best regards,
                Furniture Store Team
                """
        print(email_template)
        self.state.status = "email_sent"


def kickoff():
    sales_flow = SalesFlow()
    sales_flow.kickoff(inputs={"customer_name": "Shriya" , "order_text": "I want 5 chairs and one dining table."})
    sales_flow.plot()


if __name__ == "__main__":
    kickoff()
