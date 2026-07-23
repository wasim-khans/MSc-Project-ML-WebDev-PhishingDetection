# Dataset Inspection Report

## Source File

`machine_learning/datasets/main/raw/PhiUSIIL_Phishing_URL_Dataset.csv`

## Shape

- Rows: 235,795
- Columns: 56

## Confirmed Project Columns

- URL column: `URL`
- Label column: `label`
- Label meaning: `1` = legitimate, `0` = phishing

## Label Distribution

| Label | Meaning | Rows |
|---:|---|---:|
| 0 | phishing | 100,945 |
| 1 | legitimate | 134,850 |

## Missing Values in Project Columns

| Column | Missing values |
|---|---:|
| URL | 0 |
| label | 0 |

## Exact URL Duplicate Check

| Item | Count |
|---|---:|
| Duplicate URL rows after whitespace trimming | 425 |
| URLs that appear more than once | 425 |
| Duplicate URLs with conflicting labels | 0 |

The generated processed dataset removes exact duplicate URLs only when their labels agree. The full cleaning rule is documented in `docs/dataset_cleaning.md`.

## All Columns

```text
FILENAME
URL
URLLength
Domain
DomainLength
IsDomainIP
TLD
URLSimilarityIndex
CharContinuationRate
TLDLegitimateProb
URLCharProb
TLDLength
NoOfSubDomain
HasObfuscation
NoOfObfuscatedChar
ObfuscationRatio
NoOfLettersInURL
LetterRatioInURL
NoOfDegitsInURL
DegitRatioInURL
NoOfEqualsInURL
NoOfQMarkInURL
NoOfAmpersandInURL
NoOfOtherSpecialCharsInURL
SpacialCharRatioInURL
IsHTTPS
LineOfCode
LargestLineLength
HasTitle
Title
DomainTitleMatchScore
URLTitleMatchScore
HasFavicon
Robots
IsResponsive
NoOfURLRedirect
NoOfSelfRedirect
HasDescription
NoOfPopup
NoOfiFrame
HasExternalFormSubmit
HasSocialNet
HasSubmitButton
HasHiddenFields
HasPasswordField
Bank
Pay
Crypto
HasCopyrightInfo
NoOfImage
NoOfCSS
NoOfJS
NoOfSelfRef
NoOfEmptyRef
NoOfExternalRef
label
```

## First Five URL/Label Rows

```text
                               URL  label
  https://www.southbankmosaics.com      1
          https://www.uni-mainz.de      1
    https://www.voicefmradio.co.uk      1
       https://www.sfnmjournal.com      1
https://www.rewildingargentina.org      1
```

## Project Decision

Use only `URL` and `label` from the raw dataset, then generate fresh URL-only lexical features in project code.
