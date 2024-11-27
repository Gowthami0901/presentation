
```mermaid
sequenceDiagram
    participant User
    participant PropertyDetailComponent
    participant PropertyService
    participant BackendAPI

    User->>+PropertyDetailComponent: Navigate to Property Detail Page
    PropertyDetailComponent->>+PropertyService: Fetch property details by ID
    PropertyService->>+BackendAPI: GET /properties/:id
    BackendAPI-->>-PropertyService: Return property details
    PropertyService-->>-PropertyDetailComponent: Return property details
    PropertyDetailComponent-->>-User: Display property details
    Note over User: User views the property details
```
