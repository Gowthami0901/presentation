```mermaid
sequenceDiagram
    participant User
    participant PropertyListingComponent
    participant PropertyDetailComponent
    participant ActivatedRoute
    participant PropertyService

    User->>PropertyListingComponent: Load Property Listing
    PropertyListingComponent->>PropertyService: getProperties()
    Note right of PropertyService: Ensure proper authentication and authorization
    PropertyService-->>PropertyListingComponent: properties[]
    PropertyListingComponent-->>User: Display properties

    User->>PropertyDetailComponent: Select Property
    PropertyDetailComponent->>ActivatedRoute: Get property ID from route
    ActivatedRoute-->>PropertyDetailComponent: property ID
    PropertyDetailComponent->>PropertyService: getProperty(id)
    Note right of PropertyService: Validate and sanitize property ID
    PropertyService-->>PropertyDetailComponent: property
    PropertyDetailComponent-->>User: Display property details
    PropertyService-->>PropertyDetailComponent: error (if any)
    Note right of PropertyDetailComponent: Handle errors securely to avoid information leakage
    PropertyDetailComponent-->>User: Display error message (if any)
```
