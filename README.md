# PMU-B-PersonalAI

## Round 2 | “Personal AI Portfolio”

Know me better from this presentation video click 👉 [![Watch the video](http://img.youtube.com/vi/xmGU1ydM88Q/3.jpg)](https://youtu.be/xmGU1ydM88Q)

This repository consists of my notes from the lectures and code from the workshop.

For the workshop, I attached files of each lecture here.

- [xPore](https://github.com/pamarysl/PMU-B-PersonalAI/blob/a0397b0d4c97474d7fd9168ef6737362359136b4/XPore-Workshop_GMM.ipynb)
- [Mental disorder detection from social media data](https://github.com/pamarysl/PMU-B-PersonalAI/blob/c3192d91f2826f2b64d2b7faa81e6e79066202a7/SocialMedia-Workshop.ipynb)
- [Biosignal](https://github.com/pamarysl/PMU-B-PersonalAI/blob/945e0d09fd57c60cd27e7d17029c9d2aa960d757/Biosignal-Workshop_model.py) 
- [CodeClone](https://github.com/pamarysl/PMU-B-PersonalAI/blob/90f8ed3bd57905a740f736a30204a872fca1ed74/CodeCloneDetection-Workshop.ipynb)
- [BiTNet](https://github.com/pamarysl/PMU-B-PersonalAI/blob/686a4eea167f2e80c4bd231b8c79bc02d1463c4f/ImageClassification-Workshop_EfficientNetB5.ipynb)
- [AI for arresting criminals](https://github.com/pamarysl/PMU-B-PersonalAI/blob/1e499cb7e8ae48f7b107a43ed0d683c9771bddd2/Image-Workshop_ObjectDetection.ipynb)




For summary note, kindly read following paragraphs.
##### Table of Contents  
[xPore](#xPore)  
[Mental disorder detection from social media data](#Socialmedia)  
[Biosignal](#Biosignal)  
[CodeClone](#CodeClone)  
[BiTNet](#BiTNet)  




# xPore <a name="BiTNet"/>

### ****An AI-Powered App for Bioinformaticians****

### Introduction

xPore : AI base application to find RNA modification used in bioinformatics 

Input data : Nanopore DNA sequencing → สัญญาณไฟฟ้า

![Overall of study journal]([xPore%20249c74fc1a4f42c9a7898f4931d49a69/Untitled.png](https://www.notion.so/pam-sl/xPore-249c74fc1a4f42c9a7898f4931d49a69?pvs=4#e3e6a640d51746b9b0a504ca52e31eaa))

Overall of study journal

RNA modification : การปรับเปลี่ยน/ดัดแปลง สาย RNA ถ้ามีไม่ได้ถือว่าผิดปกติ

m6A : molecule produced from RNA modification → is focused in this study → consider if there is association between modification and disease, a study found a emzyme help detect m6A called ‘m6ACE-Seq’

Modification rates : % modified read per total read

RNA sequencing (Nanopore) : มี protein pore อยู่ แล้วเราหยด sample ที่มี DNA ลงมา ทำให้ pore สามารถดึง DNA ผ่าน pore แล้วจะเกิดเป็นค่าไฟฟ้าที่เปลี่ยนแปลงไป ซึ่งสามารถวัดได้ แล้วสามารถนำไปแปลงเพื่อดูว่าเป็น DNA อะไร จาก pattern ของกระแสไฟฟ้าที่เฉพาะของแต่ละเบส

ข้อดี : ขนาดเล็ก, Real-time seq, Real-time base call

Base calling : ชื่อกระบวนการแปลงค่าสัญญาณไฟฟ้าเป็นลำดับเบส ด้วย ML ; FAST5(อ่านค่าสัญญาณไฟฟ้าจากเบสที่ละ5ตัว หรือ 5-mer) → FASTQ(เก็บข้อมูลเบสที่callแล้ว/base sequence) หรือก็คือ FASTQ เป็นผลลัพธ์จาก Base calling

FASTA : Reference sequence เป็นข้อมูลที่ถูกต้อง เหมือนเป็น data base

BAM/SAM : Alignment result (FASTQ aligned with FASTA)

Central Dogma : DNA → mRNA → Protein

In this study focused on **mRNA →** gene expression → มาจากการ transcribe ; transcribe↑ #read↑ → use Nanopore to compare if RNA modification relate with #read 

### Nanopore pre-processing pipeline

![Untitled](xPore%20249c74fc1a4f42c9a7898f4931d49a69/Untitled%201.png)

### Problem statement

RNA → sequence in Nanopore → สัญญาณไฟฟ้า

Research objective : (1) m6A location and (2) Modification rates / quantify

ถ้า Modification rates ต่างมาก (sig) → ไปดูว่าเกี่ยวข้องกับโรคเปล่า

### Technique

**Gaussian Mixture Modelling (GMM)**

![มีหลาย distribution ผสมกัน ](xPore%20249c74fc1a4f42c9a7898f4931d49a69/Untitled%202.png)

มีหลาย distribution ผสมกัน 

ใช้สมมติฐาน(สูตร)นี้ ในการ 

- infer การมีกลุ่มใน data โดย model จะทำงานแบบ iterative เพื่อจัดกลุ่มข้อมูล (หาค่า parameter ตามสูตร GMM ที่ดีที่สุด)
- สร้างข้อมูล เนื่องจากมีช่วงของการ sampling อยู่ → Generative AI (ประเภทหนึ่งของ AI ex. AI สร้างรูป) note: image classification→ discriminative ML (อีกประเภทหนึ่งของ AI ) , Clustering, Classification

**Bayesian**

Learning algorithm สำหรับ หาค่า parameter ตามสูตร GMM ที่ดีที่สุด

![Untitled](xPore%20249c74fc1a4f42c9a7898f4931d49a69/Untitled%203.png)

**Frequentist / Point estimation**

ผลลัพธ์เป็นตัวเลขชุดเดียว (theta) ที่ทำให้ Data มี prob สูงที่สุด

![Untitled](xPore%20249c74fc1a4f42c9a7898f4931d49a69/Untitled%204.png)

![Untitled](xPore%20249c74fc1a4f42c9a7898f4931d49a69/Untitled%205.png)

**Apply Bayesian Multi-Sample Gaussian Mixture Modelling in bioinforamatics**

มีกระแสไฟจาก RNA seq ทั้งหมด 3 ตัวอย่าง →  **Multi-Sample** (มีกี่ตัวอย่างก็ได้เท่าที่คอมรับไหว) → เห็นส่วนที่ข้อมูลมีความแตกต่างกันระหว่าง 3 ตัวอย่าง → เอามา plot เป็น distribution → **fit Gaussian** (find param of the dist) กรณีนี้มี 2 dist คือ Modified&Unmodified → ได้ 2 dist ที่มีข้อมูลกระจายตัวอยู่ → ใช้ **Bayesian** หา prob

Note: 

- know unmodified from database → have Prior in **Bayesian** formula
- Nanopolish use **Gaussian** dist as assumption

**Speed-up ML** 

- Built Config file
    
    ![Untitled](xPore%20249c74fc1a4f42c9a7898f4931d49a69/Untitled%206.png)
    
    - ในโค้ด ต้องเขียน script เพื่ออ่านไฟล์ config ด้วย
- Python packaging
    - เริ่มวางโครงสร้างตั้งแต่แรกเริ่ม project
    - เวลาใช้ **ใช้คำสั่ง import ได้เลย** โดยไม่ต้องเรียก path ใดๆให้ยุ่งยาก คนอื่น เครื่องอื่น ก็เรียนผ่าน import ได้เลย
- Parallelization
- File indexing

### Evaluation

Validation 

- consume most time of study (ML<preprocess<validation)
- use to analyze the results to gain insight (good, weak point of used method)

Compare with other state-of-the-art methods

Application

- explain how to use w/ external data
- have ‘Human evaluation’ in late of study will gain more reliability
- explain what we discover from this study

### Visualization & Presentation

![each fig contains sub fig/graph](xPore%20249c74fc1a4f42c9a7898f4931d49a69/Untitled%207.png)

each fig contains sub fig/graph

Storyline

- focusing on method → start w/ method overview
- validation of 2 objectives
- Application & discovery

Choose the right plots → image is easy to understand

- plot w/ Python → edit/compose w/ PPT or PS
- understandable & meaningful > beautiful

![2. → นำไปประยุกต์ใช้กับอย่างอื่นได้เยอะ และมีการประเมินที่น่าเชื่อถือ](xPore%20249c74fc1a4f42c9a7898f4931d49a69/Untitled%208.png)

2. → นำไปประยุกต์ใช้กับอย่างอื่นได้เยอะ และมีการประเมินที่น่าเชื่อถือ

### Future Work

consider of

- limitations of the study
- change in future that can make our approach not work




------------------------------------------------------------------------------------------




# Social media <a name="Socialmedia"/>

about NLP

professor email: [wongkoblap@sut.ac.th](mailto:wongkoblap@sut.ac.th)

- crawling from Twitter max 3,200 tweets.
- Transformer model / Deep learning → use more data to train due to more complexity
- keyword : bigram ก็เพียงพอแล้วในการใช้สร้างโมเดล วิเคราะห์ข้อมูล จริงๆก็ต้องทดลองและเปรียบเทียบดูว่าแบบไหนดีกว่า

### Problem Statement

social media เป็นช่องทางหนึ่งที่คนมาแสดงความรู้สึกของตัวเอง → ถ้าเราสามารถเข้าใจถึงความรู้สึกเหล่านั้นและระบุถึงภาวะซึมเศร้าได้โดยใช้ข้อมูลจาก social media → ประโยชน์ต่อสังคม

สาเหตุที่คนชอบใช้ social media : เข้าถึงได้ทันที, มีข้อมูลให้ดูมากมาย, จะโพสอะไรก็ได้, มีฟีดที่สอดคล้องกับความสนใจ

### Data Collection

- Directly collect from human ex. questionaires, EHR (working w/ hospital)
- Aggregate from public → need to annotate the data (lebel)
- Open datasets
- EHR (working w/ hospital)

### Data Exploration & Preprocessing

![Untitled](Social%20media%20c7427e3be563489383ed5550a5ab2a61/Untitled.png)

Apart from Computer science, We need to have **Domain knowledge** too ex. understanding in interested disease 

**Future Extraction** ex. CountVectorizer(), social media consuming time

**Visualization** ex. compare the word using between normal & depress

### Predictive Modeling

NLP w/ supervised learning

![Untitled](Social%20media%20c7427e3be563489383ed5550a5ab2a61/Untitled%201.png)

### Model Evaluation

Method to evaluate the performance of the model ex. look at accuracy to decide how many features should be included → many features aren’t always good 

ex. AUC



------------------------------------------------------------------------------------------




# Biosignal <a name="Biosignal"/>

### Introduction

Step in Biosignal analysis

(1) **Preprocessing** (remove noise) → (2) **Feature extraction** (derived features that related w/ disease, usually use specialist to label (Hand-engineered)) → (3) **Model construction**

Step (1) & (2) are Application-Specific step → slove w/ “**Deep Learning**”

### Sleep Stage Scoring

วัดประสิทธิภาพการนอน

Collect and score Polysomnogram (PSG) ประกอบด้วยสัญญาณ ดังนี้  EEG, EOG, ECG, EMG

![5 sleep stages; N1, N2, N3, REM, W](Biosignal%20d358467189514a5b8e0366b4192bb2c4/Untitled.png)

5 sleep stages; N1, N2, N3, REM, W

### Problems

- การแปลผลสัญญาณใช้เวลานาน และต้องอาศัยผู้เชี่ยวชาญ
- จน.ผู้ป่วยเพิ่มสูงขึ้น
- ไม่สามารถติดตั้งอุปกรณ์ที่บ้านได้ สัญญาณอาจจะไม่มีคุณภาพ และจะแปลผลไม่ได้

### Models

**DeepSleepNet**

- model for sleep stage scoring
- Single-channel EEG (30 วิ)
- 

![Untitled](Biosignal%20d358467189514a5b8e0366b4192bb2c4/Untitled%201.png)

โครงสร้าง ประกอบด้วย CNN large(filterใหญ่/ยาว), small (filterเล็ก/สั้น) เนื่องจากต้องการสกัด feature จากสัญญาณที่ทั้งมีความถี่สูงและต่ำ (ใช้ CNN large) อีกทั้งตัว CNN เรียนรู้เองจากข้อมูลและหา pattern เอง ไม่เหมือนแต่ก่อนที่มีความรู้อยู่แล้วว่าคลื่นสมองเหมือนกราฟ sine ทำให้ CNN สามารถปรับความจำเพาะในผู้ป่วยแต่ละรายได้

![Untitled](Biosignal%20d358467189514a5b8e0366b4192bb2c4/Untitled%202.png)

ส่วนต่อมาเป็นโครงสร้างแบบ RNN เนื่องจากในการตัดสินมีการดู condition ก่อนหน้าร่วมด้วย ตามทฤษฎีทางการแพทย์ แต่ก็มีเส้นทางลัดที่ไม่ได้ผ่าน RNN ให้โมเดลตัดสินใจเลือกใช้ด้วย

![Untitled](Biosignal%20d358467189514a5b8e0366b4192bb2c4/Untitled%203.png)

Aim: find θ that mininize summation

**TinySleepNet**

- Improve of **DeepSleepNet →** smaller but work same/better
- Don’t have CNN large, small but has more filter
- Have RNN but less layer
- Need different training methods: Data augmentation/Sequence augmentation, Weighted cross-entropy loss (prevent overfit when class not balanced), No pre-training

### Evaluation

![Untitled](Biosignal%20d358467189514a5b8e0366b4192bb2c4/Untitled%204.png)

Performance matrics

- overall ดูความแม่นยำโดยรวมของทั้ง 5 class
    - ปกติจะรายงานทั้ง confusion metrics แต่หลักๆจะดูที่ accuracy, F1 เป็นหลักเพราะสามารถอธิบายประสิทธิภาพโดยรวมได้
    - F1 จะค่าสูงได้เมื่อทั้ง precision, recall สูงเท่านั้น → ดูภาพรวมได้ดี
- Per-class **ดูว่าโมเดลทำนาย class ไหนได้ดีเป็นพิเศษ**

K-fold cross-validation

- ในการ Train model ด้านการแพทย์ มักไม่แบ่ง train test ปกติ แต่จะใช้ cross-validation ในการหา best parameter ก่อนไป test
- model will be learned from all dataset, for test set usually use another dataset

Hypnogram ฮิปโนแกรม

![Untitled](Biosignal%20d358467189514a5b8e0366b4192bb2c4/Untitled%205.png)

### Future work

- DL เหมาะกับงานที่เป็น supervised learning ที่มีผู้เชี่ยวชาญ label ให้มากกว่า
- alternatively, เราจะแปลง signal → spectrogram (image) → use CNN ก็ได้ แต่มูลบางส่วนจะหายไป
- สามารถนำโมเดลไปใส่ใน wearable device ได้ เพราะใช้ Pytorch มี library สำเร็จ

### Assignment

- แก้โค้ดใน [model.py](http://model.py) → def simple_model() ให้มีโครงสร้างเหมือน **TinySleepNet** ในส่วน representation learning part

![แก้ส่วนที่คลุม](Biosignal%20d358467189514a5b8e0366b4192bb2c4/Untitled%206.png)

แก้ส่วนที่คลุม

![Untitled](Biosignal%20d358467189514a5b8e0366b4192bb2c4/Untitled%207.png)

- แก้ in_features=384 ต้องเป็นเลขอื่นเนื่องจากเราเปลี่ยน layer → ลองรันแล้วดูโค้ด error
- ทำแล้วมีปัญหาเรื่อง kernel size ใหญ่กว่า feature → เพิ่ม padding




------------------------------------------------------------------------------------------



# CodeClone  <a name="CodeClone"/>

AI for detecting code plagiarism

### Introduction

![Untitled](CodeClone%206299c4f55a01416fa2a86f0deda3133b/Untitled.png)

![Untitled](CodeClone%206299c4f55a01416fa2a86f0deda3133b/Untitled%201.png)

Type

- 1 เหมือนเป๊ะ
- 2 เปลี่ยนชื่อตัวแปร
- 3 เพิ่มลดบรรทัด
- 4 หน้าที่เหมือนกัน แต่เขียนไม่เหมือน → แยกยากที่สุด

![Untitled](CodeClone%206299c4f55a01416fa2a86f0deda3133b/Untitled%202.png)

### Problem statement

- เครื่องมือ/วิธีการในปจบ.ยังตรวจ source code ที่ถูก modifications มากๆไม่ได้
- เครื่องมือ/วิธีการในปจบ. เป็น command line base → ใช้งานยาก

### Objectives

![Untitled](CodeClone%206299c4f55a01416fa2a86f0deda3133b/Untitled%203.png)

### Method

Model → Decision tree, Random forest, SVM, SVM + SMO

Dataset → BigCloneBench(BCB) ถูกใช้อย่างแพร่หลายในงานวิศวกรรมซอฟแวร์ด้าน codeclone

Present → web-based style with UI

สามารถเชื่อมต่อและดึงโค้กจาก Github

### Modeling

1. **Building** Merry Engine

Code Metrics Extraction

- Syntactic metrics: ดูจาก systec ของภาษาต่างๆ ex. #variable, #line
- Semantic metrics: ดูหลักการทำงานของโค้ด ใช้ code2vec (pretrain neural model) แปลงโค้ดให้เป็น vector → cosine similarity วัดความต่าง
1. **Using** Merry Engine for Clone Detection
    
    ![Untitled](CodeClone%206299c4f55a01416fa2a86f0deda3133b/Untitled%204.png)
    

### Evaluation

1. ความแม่นยำใน dataset → ใช้ precition, recall, F1

precision บอกว่าหยิบมา 100 ถูกกี่% recall บอกว่าจากที่ถูก 100 หยิบมาได้กี่%

![Untitled](CodeClone%206299c4f55a01416fa2a86f0deda3133b/Untitled%205.png)

แต่ละโมเดล การใช้ Syntactic + Semantic ให้ผลดีกว่าเสมอ

![Untitled](CodeClone%206299c4f55a01416fa2a86f0deda3133b/Untitled%206.png)

1. ความแม่นยำใน real project

ดึงโค้ดมาเข้าโมเดลแล้วให้ผู้มีความรู้javaลงความเห็นว่าเห็นด้วยกับที่โมเดลทำนายมั้ย

1. การใช้งาน

![Untitled](CodeClone%206299c4f55a01416fa2a86f0deda3133b/Untitled%207.png)

ให้กลุ่มตัวอย่างดูวิดิโอการใช้งานเครื่องมือที่มีอยู่เทียบกับเครื่องมือที่เราคิดค้น

### Future work

- ลองพัฒนากับภาษาโค้ดอื่น
- เพิ่มจน.กลุ่มตัวอย่างที่ทดลองใช้
- ลดเวลาประมวลผล
- นำไปใช้ในการเรียนการสอนคอมพิวเตอร์

Assignment

example

![Untitled](CodeClone%206299c4f55a01416fa2a86f0deda3133b/Untitled%208.png)






------------------------------------------------------------------------------------------





# BiTNet  <a name="xPore"/>

AI for diagnosing ultrasound image

### Data

**Challenge**

- the image quality depends on performer

![Untitled](BiTNet%20d796da69ab994fdabee6d6e9cb8c9f78/Untitled.png)

- High workload for specialist

**Preprocessing**

1. จัดข้อมูลให้อยู่ในโฟลเดอร์ตาม class และตั้งชื่อที่แสดงถึง class,case name
2. สร้าง metadeta ของแต่ละรูป
3. remove background
    
    ![Untitled](BiTNet%20d796da69ab994fdabee6d6e9cb8c9f78/Untitled%201.png)
    
4. Check the input size for the chosen model
5. data augmentation →Horizontal Shift, Vertical Shift, Rotation 30°, random brightness, shear ดึงภาพให้เบี้ยว, zoom

### Modelling

![Untitled](BiTNet%20d796da69ab994fdabee6d6e9cb8c9f78/Untitled%202.png)

**EfficientNet**

- developed by Google
- included in TensorFlow (library by Google)
- มีหลายขนาด
    
    ![Untitled](BiTNet%20d796da69ab994fdabee6d6e9cb8c9f78/Untitled%203.png)
    
- problem: โมเดลให้ความมั่นใจสูงในข้อมูลที่ทายผิด → ความมั่นใจสูงจะชักจูงแพทย์ให้คล้อยตาม → slove w/ Random forest
    
    ![Untitled](BiTNet%20d796da69ab994fdabee6d6e9cb8c9f78/Untitled%204.png)
    

**BiTNet**

![Untitled](BiTNet%20d796da69ab994fdabee6d6e9cb8c9f78/Untitled%205.png)

### Application

มี 2 แอพ

1. For pre-screening: model filters the image for make priority of the doctor access and diagnose → low workload
2. For assist: tell if the input image show abnormal/disease (15 classes) and locate the lesion (use Gradcam)
