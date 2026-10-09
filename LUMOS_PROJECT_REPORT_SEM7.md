# ANATOMY-GUIDED MULTIVIEW MULTITASK DEEP LEARNING FOR LUMBAR BONE MINERAL DENSITY ESTIMATION AND OSTEOPOROSIS SEVERITY ASSESSMENT FROM X-RAY IMAGES

## 22CSP72 – PROJECT WORK II PHASE I - A PROJECT REPORT

**Submitted by:**
- AADHI PRANESH S S (23CSR001)
- ABHINAV KRISHNA B (23CSR005)
- DHEEPISHA G (23CSR050)

**Department of Computer Science and Engineering, Kongu Engineering College (Autonomous), Perundurai, Erode – 638 060**
**Anna University, Chennai — October 2026**

---

## TABLE OF CONTENTS

- ** ABSTRACT** (Page v)
- ** LIST OF TABLES** (Page viii)
- ** LIST OF FIGURES** (Page ix)
- **1. INTRODUCTION** (Page 1)
- **1.1 Background and Importance of Bone Health Screening** (Page 1)
- **1.2 Problem Statement** (Page 3)
- **1.3 Objectives of the Work** (Page 4)
- **1.4 Scope and Expected Deliverables** (Page 5)
- **2. LITERATURE SURVEY** (Page 6)
- **2.1 Review of State-of-the-Art Deep Learning in Bone Health** (Page 6)
- **2.2 Comparative Analysis of Selected Research Works** (Page 8)
- **2.3 Research Gaps and Proposed Novelties** (Page 10)
- **3. EXISTING SYSTEM** (Page 12)
- **3.1 Conventional Clinical Diagnostic Modalities** (Page 12)
- **3.2 Prior AI-Based Radiographic Approaches** (Page 13)
- **3.3 Inherent Limitations of Existing Systems** (Page 14)
- **3.4 Need for an Improved Anatomy-Guided Multitask Framework** (Page 15)
- **4. REQUIREMENT SPECIFICATION** (Page 17)
- **4.1 Functional Requirements** (Page 17)
- **4.2 Non-Functional Requirements** (Page 19)
- **4.3 Hardware Requirements** (Page 20)
- **4.4 Software Requirements and Environment** (Page 21)
- **4.5 Clinical User and Workflow Requirements** (Page 22)
- **5. PROPOSED SYSTEM ARCHITECTURE & METHODOLOGY** (Page 24)
- **5.1 End-to-End Architectural Overview** (Page 24)
- **5.2 Multimodal Dataset Preprocessing & Patient Linkage** (Page 26)
- **5.3 Anatomy-Guided Vertebral Localization & ROI Extraction** (Page 28)
- **5.4 Dual-Branch Multiview Multitask Deep Learning Network** (Page 30)
- **5.5 Inter-Vertebral Transformer Modeling** (Page 32)
- **5.6 Stochastic Missing-View Robust Fusion** (Page 33)
- **5.7 Multitask Loss Formulation & Joint Optimization** (Page 34)
- **5.8 Anatomically Grounded Explainability (Grad-CAM)** (Page 35)
- **5.9 Clinical Web Platform & API Architecture** (Page 36)
- **6. RESULTS AND DISCUSSION** (Page 38)
- **6.1 Experimental Setup & Implementation Details** (Page 38)
- **6.2 Vertebral Segmentation & ROI Extraction Performance** (Page 39)
- **6.3 Continuous BMD Regression Performance Across Targets** (Page 40)
- **6.4 Three-Tier Osteoporosis Severity Classification Results** (Page 42)
- **6.5 Visual Explainability & Radiographic Heatmap Inspection** (Page 44)
- **6.6 Clinical Web Application Demonstration & Usability** (Page 46)
- **6.7 Comparative Discussion & Diagnostic Implications** (Page 47)
- **7. CONCLUSION AND FUTURE WORK** (Page 49)
- **7.1 Conclusion** (Page 49)
- **7.2 Future Scope and Clinical Enhancements** (Page 50)
- ** APPENDIX 1: CORE ALGORITHMIC IMPLEMENTATION** (Page 52)
- ** APPENDIX 2: CLINICAL INTERFACE ARTIFACTS** (Page 55)
- ** REFERENCES** (Page 58)
- ** MAPPING TO SUSTAINABLE DEVELOPMENT GOALS (SDGs)** (Page 61)

---

## ABSTRACT

Osteoporosis is a prevalent systemic skeletal disease characterized by low bone mass, micro-architectural degradation of bone tissue, and elevated fragility fracture risk... [See full report in DOCX]

## SUMMARY OF CHAPTERS

- **Chapter 1: Introduction**: Background, Problem Statement, Objectives, Scope
- **Chapter 2: Literature Survey**: State-of-the-art review, Table 2.1 comparative analysis, Research Gaps
- **Chapter 3: Existing System**: Clinical DXA/QCT modalities, prior AI radiographic classifiers, inherent limitations
- **Chapter 4: Requirement Specification**: Functional, Non-functional, Hardware, Software, User requirements
- **Chapter 5: Proposed System**: Architecture (Fig 5.1), Multimodal ingestion, SAM localization, Inter-Vertebral Transformer, Stochastic view fusion, Multitask loss formulation, Grad-CAM explainability, Web platform
- **Chapter 6: Results and Discussion**: 120-patient held-out test cohort results (MAE: 0.1508 g/cm², Macro AUC: 0.6992), Confusion matrix, Grad-CAM visual verification, Web interface performance
- **Chapter 7: Conclusion & Future Work**: Key conclusions, 3D CT integration, longitudinal fracture risk modeling
- **Appendix 1 & 2**: Algorithmic implementation & UI artifacts
- **References & SDG Mapping**: IEEE academic citations & UN SDG 3, 9, 10 alignment
