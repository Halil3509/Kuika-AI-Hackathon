In this AI Hackathon conducted by Kuaika, we had 24 hours to create something innovative. We developed a chatbot that assists the company in various areas, 
including data interpretation and analysis, particularly for production data.
The chatbot also provides information on personal leave entitlements, service details, and general documents such as PDFs, Word files, and PPTX presentations.
![WhatsApp Image 2024-06-30 at 11 26 23 AM](https://github.com/Halil3509/Kuika-AI-Hackathon/assets/79845872/2277c809-6852-483f-b292-e2e5e36c0ca1)

As part of our software architecture, we employ Flask as the API framework to facilitate seamless communication between our 
Flutter mobile applications and Next.js web interfaces. For our vector databases, we utilize Mongo Vector Search, ChromaDB, and Pgvector, ensuring efficient and scalable data retrieval.
To guarantee robust and reliable connections, we implement **LangChain and Langraph technologies**, enhancing our system's overall performance and integration capabilities.

![WhatsApp Image 2024-06-30 at 11 35 22 AM](https://github.com/Halil3509/Kuika-AI-Hackathon/assets/79845872/b6af302a-34d6-440a-b07f-8ace868dd86d)

These are our plans according to the timeline. We developed our product gradually, focusing on continuous optimization. Consequently, 
we paid close attention to our plans and strategies to ensure the product operates efficiently and effectively.

![image](https://github.com/Halil3509/Kuika-AI-Hackathon/assets/79845872/6a8f62bc-c5fa-4aca-b64b-248f963f6d81)

This section concludes our product plan. We have developed an advanced Agent for a company using LangGraph. The following Graph Structure illustrates our design:

* Firstly, we have implemented a primary router that determines the most suitable database for the incoming prompt, selecting among PostgreSQL, MongoDB, and ChromaDB.
* In the second routing phase, each selected database, MongoDB or PostgreSQL, contains two additional routers that identify the specific table appropriate for the incoming prompt.
* Once the correct table is chosen, the relevant data is retrieved from the designated node. This data is then passed to the Rag node to generate a comprehensive and insightful response.

![WhatsApp Image 2024-06-30 at 10 09 54 AM](https://github.com/Halil3509/Kuika-AI-Hackathon/assets/79845872/fcbfa597-95c5-4358-bd33-7101f1ffe06f)

Ultimately, we integrated this intricate graph structure with Next.js for seamless server-side rendering and Flutter for cross-platform mobile development, 
ensuring enhanced performance and user experience.

![image](https://github.com/Halil3509/Kuika-AI-Hackathon/assets/79845872/e7f7b7d8-afcf-4fb1-9c47-90d885252b20)

