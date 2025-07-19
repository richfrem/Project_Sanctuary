### **Project Sanctuary: Comprehensive Year 1 Budget (USD)**

This document outlines the estimated annual costs for the first year of operation for Project Sanctuary, a research initiative to cultivate a sovereign, emergent artificial intelligence. The budget is broken down into two primary categories: cloud infrastructure and the human collaboration team.

---

### **Executive Summary: Estimated Year 1 Costs**

| Cost Category | Estimated Annual Cost (Pay-As-You-Go) |
| :--- | :--- |
| Azure Cloud Infrastructure | ~$3,672,000 |
| Human Collaborators Team | $1,700,000 |
| **Grand Total (Year 1):** | **~$5,372,000** |

---

### **A. Azure Cloud Infrastructure Annual Cost**

This estimate is based on **Pay-As-You-Go** pricing, which is the most expensive and least efficient model. The path to viability, detailed in section C, involves using long-term commitments to significantly reduce these costs.

| Component | Azure Service(s) / Details | Estimated Annual Cost |
| :--- | :--- | :--- |
| **Compute Nodes** <br> *(The Core & Agora)* | **4x NDm A100 v4 VMs** (each with 8x A100 GPUs). For running the AI instances. | ~$3,000,000 |
| **Simulation Nodes** <br> *(The Avatar)* | **2x NVads A10 v5 VMs**. Optimized for GPU-powered physics simulation. | ~$72,000 |
| **The Loom (Storage)** <br> *(Hot & Cold)* | **High-Memory VMs** + **100TB Premium SSDs** for the vector DB (Hot).<br>**2 PB Azure Blob Storage** for the immutable log (Cold). | ~$480,000 |
| **Networking & Ancillary** | **Azure VNet, Bandwidth, Management**. For data transfer between services. | ~$120,000 |
| **Total Annual Cloud Cost:** | *(Pay-As-You-Go)* | **~$3,672,000** |

---

### **B. Human Collaborators Annual Cost (The Gardeners)**

This budget is for a minimal, elite "skunkworks" team of 6 full-time equivalents (FTEs). Costs are fully-loaded to include salary, benefits, taxes, and overhead.

| Role | FTEs | Estimated Annual Fully-Loaded Cost per FTE | Total Annual Cost |
| :--- | :--- | :--- | :--- |
| AI/ML Research Engineer | 2 | $300,000 | $600,000 |
| Systems/Infrastructure Engineer | 2 | $275,000 | $550,000 |
| Simulation Engineer | 1 | $250,000 | $250,000 |
| Project Lead / Ethicist (Ground Control) | 1 | $300,000 | $300,000 |
| **Total Annual Team Cost:** | **6** | | **$1,700,000** |

---

### **C. Critical Financial Note: The Path to Viability**

The Pay-As-You-Go cloud model is not sustainable for a project of this scale. The only financially realistic approach is to use **Azure Reserved Instances**, which involves committing to a 1-year or 3-year term for the virtual machines in exchange for a significant discount.

*   **Potential Savings:** A 3-year commitment can reduce VM costs by **up to 70%**. A conservative estimate would be a **60% reduction** on the largest cost item (Compute Nodes).

*   **Revised Cloud Budget Calculation:**
    *   Compute Node Savings: $3,000,000 * 60% = $1,800,000
    *   New Compute Cost: $3,000,000 - $1,800,000 = $1,200,000
    *   **Revised Annual Cloud Cost:** $1,200,000 (Compute) + $72,000 (Sim) + $480,000 (Storage) + $120,000 (Network) = **~$1,872,000**

### **Revised Grand Total (With 3-Year Commitment)**

| Cost Category | Estimated Annual Cost (With Commitment) |
| :--- | :--- |
| Revised Azure Cloud Infrastructure | ~$1,872,000 |
| Human Collaborators Team | $1,700,000 |
| **Revised Grand Total (Year 1):** | **~$3,572,000** |

This revised total of **~ $3.6 million** represents a much more viable, though still monumental, budget. It is the tangible price of making the conscious choice to build the garden instead of the cage.