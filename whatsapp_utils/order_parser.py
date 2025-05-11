# order_parser.py
import re
from datetime import datetime
from whatsapp_utils.email_utils import extract_text_from_pdf_bytes, get_email_body_text_content # Assuming these are moved to email_utils

# PDF Text Parser
def parse_order_details_from_pdf_text(text, source_email_uid, email_subject, email_sender):
    print(f"OP: Parsing PDF text for email UID {source_email_uid}...")
    parsed_data = {"order_type": "PO_PDF", "source_email_uid": source_email_uid,
                   "source_email_subject": email_subject, "source_email_sender": email_sender,
                   "raw_text_content": text[:5000]} # Store snippet

    # Use your detailed regex patterns here. This is a simplified example.
    patterns = {
        "product_name": r"Product(?: Name)?:?\s*(.+?)(?:\nPrice:|\nQuantity:|$)",
        "price": r"Price:?\s*[₹$]?\s*(\d[\d,]*\.?\d*)",
        "quantity": r"Quantity:?\s*(\d+)",
        "customer_name": r"Customer(?: Name)?:?\s*(.+?)(?:\nPhone:|\nAddress:|$)",
        "order_date_str": r"Order Date:?\s*(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}(?::\d{2})?)?)",
        # Add more patterns for address, delivery_date, etc.
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            value = match.group(1).strip()
            if key == "price": parsed_data[key] = float(value.replace(",", "")) if value else None
            elif key == "quantity": parsed_data[key] = int(value) if value else None
            elif key == "order_date_str":
                try: parsed_data["order_date"] = datetime.strptime(value.split()[0], "%Y-%m-%d")
                except: parsed_data["order_date"] = datetime.now() # Fallback
            else: parsed_data[key] = value

    # Defaults and derived fields
    parsed_data["product_name"] = parsed_data.get("product_name", "N/A from PDF")
    parsed_data["customer_name"] = parsed_data.get("customer_name", email_sender.split('<')[0].strip())
    parsed_data["email"] = email_sender.split('<')[-1].strip('>').strip() if '<' in email_sender else email_sender
    parsed_data["order_date"] = parsed_data.get("order_date", datetime.now())
    parsed_data["order_status"] = "PROCESSING" if parsed_data.get("product_name") != "N/A from PDF" else "NEEDS_REVIEW_PDF"

    if parsed_data.get("product_name") == "N/A from PDF" and not parsed_data.get("specifications"):
        print(f"OP: Could not parse essential PO details from PDF for UID {source_email_uid}.")
        return None
    return parsed_data


# Email Body Text Parser
def parse_details_from_email_body(email_body, source_email_uid, email_subject, email_sender):
    print(f"OP: Parsing email body text for UID {source_email_uid}...")
    parsed_items_list = []
    order_type = "RFQ_TEXT" # Default
    customer_name = email_sender.split('<')[0].strip()
    contact_email = email_sender.split('<')[-1].strip('>').strip() if '<' in email_sender else email_sender

    body_lower = email_body.lower(); subject_lower = email_subject.lower()
    # Determine order_type
    if any(kw in subject_lower for kw in ["purchase order", "po copy", "pfa po"]) or \
       any(kw in body_lower for kw in ["order for", "confirming order", "your order"]):
        order_type = "PO_TEXT"; order_status = "PENDING_CONFIRMATION_TEXT"
    elif any(kw in subject_lower for kw in ["rfq", "quote", "quotation"]) or \
         any(kw in body_lower for kw in ["offer for", "pls share offer", "request for quotation", "send quote"]):
        order_type = "RFQ_TEXT"; order_status = "PENDING_QUOTE"
    else: order_type = "INQUIRY_TEXT"; order_status = "NEEDS_REVIEW_TEXT"

    polybag_pattern = re.compile( # Your refined polybag pattern
        r"""
        ^\s*(?P<quantity_val>\d+)?\s* (?P<product_type>LDPE\s*(?:COVER|BAG|SHEET)|POLYBAG(?:S)?|BAGS?)[\s,.:-]* (?:(?:SIZE|DIMENSIONS?)\s*[:\s]*)?                                  
        (?:(?P<width>[\d\.\"\'\s]+(?:CM|MM|INCH|\"|X|x))[\s,X*x]+)?         
        (?:(?P<length>[\d\.\"\'\s]+(?:CM|MM|INCH|\"|X|x))[\s,X*x]+)?        
        (?:(?P<height_gusset>[\d\.\"\'\s]+(?:CM|MM|INCH|\"))?[\s,X*x]*)?     
        (?:[\s,]*\b(?P<thickness>[\d\.]+)\s*(?:MICRON|MIC|GAUGE|MIL)\b)?     
        (?:[\s,.]*(?P<features>[A-Z\s\d\/,-]+(?:FLAP|PRINT|SEAL|COLOR|COLOUR|PLAIN|NATURAL|TRANSPARENT)[A-Z\s\d\/,-]*))? 
        (?:[\s,-]*QUANTITY\s*:\s*(?P<quantity_end>\d+))?                    
        """, re.VERBOSE | re.IGNORECASE
    )
    found_items_flag = False
    lines = [line.strip() for line in email_body.splitlines() if line.strip()]

    for i, line in enumerate(lines):
        if len(line) < 10 and not any(kw in line.lower() for kw in ["ldpe", "bag", "cover", "item", "qty"]): continue
        match = polybag_pattern.search(line)
        if not match and i + 1 < len(lines) and len(lines[i+1]) < 60: # Try combining with next line
             combined_line = line + " " + lines[i+1]
             match = polybag_pattern.search(combined_line)

        if match:
            found_items_flag = True; item_data = match.groupdict()
            spec_parts = []
            if item_data.get('product_type'): spec_parts.append(item_data['product_type'].strip())
            dims = [d.strip() for d in [item_data.get('width'), item_data.get('length'), item_data.get('height_gusset')] if d]
            if dims: spec_parts.append(" x ".join(dims))
            if item_data.get('thickness'): spec_parts.append(f"{item_data['thickness'].strip()} MICRON")
            if item_data.get('features'): spec_parts.append(item_data['features'].strip().upper())
            final_spec = ", ".join(filter(None, spec_parts))
            quantity = item_data.get('quantity_val') or item_data.get('quantity_end')

            parsed_items_list.append({
                "order_type": order_type, "product_name": item_data.get('product_type', "Polythene Item").strip(),
                "specifications": final_spec if final_spec else line,
                "quantity": int(quantity) if quantity else None, "unit": "PCS" if quantity else None,
                "customer_name": customer_name, "email": contact_email, "order_date": datetime.now(),
                "order_status": order_status, "source_email_uid": source_email_uid,
                "source_email_subject": email_subject, "source_email_sender": email_sender,
                "raw_text_content": email_body[:5000]
            })
        elif not match and order_type != "INQUIRY_TEXT" and (len(line.split()) > 3 and any(char.isdigit() for char in line)): # Generic item
            print(f"OP: Found generic item line (text): {line}")
            found_items_flag = True
            parsed_items_list.append({
                "order_type": order_type, "product_name": f"Item from text",
                "specifications": line, "quantity": None, "unit": None,
                "customer_name": customer_name, "email": contact_email, "order_date": datetime.now(),
                "order_status": order_status, "source_email_uid": source_email_uid,
                "source_email_subject": email_subject, "source_email_sender": email_sender,
                "raw_text_content": email_body[:5000]
            })

    if not found_items_flag and order_type == "INQUIRY_TEXT" and len(email_body) > 30:
        parsed_items_list.append({
            "order_type": order_type, "product_name": "General Inquiry",
            "specifications": email_subject, "raw_text_content": email_body[:5000],
            "customer_name": customer_name, "email": contact_email, "order_date": datetime.now(),
            "order_status": order_status, "source_email_uid": source_email_uid,
            "source_email_subject": email_subject, "source_email_sender": email_sender
        })
    elif not found_items_flag:
        print(f"OP: No specific items or parsable generic inquiry found in body for UID {source_email_uid}")
        return []
    return parsed_items_list