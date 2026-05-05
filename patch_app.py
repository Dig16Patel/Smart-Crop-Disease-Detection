import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add session state vars
content = content.replace(
    'if "auth_page" not in st.session_state:\n    st.session_state.auth_page = "login"  # "login" or "register"',
    'if "auth_page" not in st.session_state:\n    st.session_state.auth_page = "login"  # "login" or "register"\n\nif "current_scan_result" not in st.session_state:\n    st.session_state.current_scan_result = None\nif "chat_history" not in st.session_state:\n    st.session_state.chat_history = []\nif "scan_analyzed" not in st.session_state:\n    st.session_state.scan_analyzed = False'
)

# 2. Update prediction logic
old_block = """        if uploaded_file is not None and predict_clicked:
            with st.spinner("🧠 AI is analyzing the image..."):
                try:"""

new_block = """        if uploaded_file is None:
            st.session_state.scan_analyzed = False
            st.session_state.current_scan_result = None
            st.session_state.chat_history = []
        elif predict_clicked:
            st.session_state.scan_analyzed = True

        if uploaded_file is not None and st.session_state.scan_analyzed:
            if predict_clicked or st.session_state.current_scan_result is None:
                with st.spinner("🧠 AI is analyzing the image..."):
                    try:
                        processed_image = preprocess_image(image)
                        model = load_model()
                        class_indices = load_class_indices()
                        class_names = {v: k for k, v in class_indices.items()}

                        predictions = model.predict(processed_image)
                        idx = np.argmax(predictions)
                        name = class_names[idx]
                        conf = float(predictions[0][idx]) * 100
                        info = get_recommendation(name)

                        severity = info["severity"]
                        sev_cls, sev_score = "sv-none", 0
                        sev_card = "green"
                        if severity == "Low":
                            sev_cls, sev_score, sev_card = "sv-low", 25, "amber"
                        elif severity == "Moderate":
                            sev_cls, sev_score, sev_card = "sv-mod", 50, "amber"
                        elif severity == "High":
                            sev_cls, sev_score, sev_card = "sv-high", 85, "red"

                        display_name = name.replace("_", " ")

                        # Auto-save scan to database
                        save_scan(
                            user_id=st.session_state.user["id"],
                            disease_name=display_name,
                            confidence=round(conf, 2),
                            severity=severity,
                            latitude=map_lat if share_location else None,
                            longitude=map_lon if share_location else None
                        )
                        
                        st.session_state.current_scan_result = {
                            "name": name, "display_name": display_name, "conf": conf,
                            "info": info, "severity": severity, "sev_cls": sev_cls,
                            "sev_score": sev_score, "sev_card": sev_card
                        }
                        st.session_state.chat_history = []
                    except Exception as e:
                        st.error(f"⚠️ Error classifying: {e}")
                        st.session_state.scan_analyzed = False
                        
            if st.session_state.current_scan_result is not None:
                res = st.session_state.current_scan_result
                name = res["name"]
                display_name = res["display_name"]
                conf = res["conf"]
                info = res["info"]
                severity = res["severity"]
                sev_cls = res["sev_cls"]
                sev_score = res["sev_score"]
                sev_card = res["sev_card"]
                try:"""

content = content.replace(old_block, new_block)

# 3. Add chatbot UI after PDF generation
old_pdf_block = """                    except Exception as pdf_err:
                        st.warning(f"Could not generate PDF: {pdf_err}")

                except Exception as e:
                    st.error(f"⚠️ Error: {e}")
        else:"""

new_pdf_block = """                    except Exception as pdf_err:
                        st.warning(f"Could not generate PDF: {pdf_err}")
                        
                    # 🤖 AI Agronomist Chatbot
                    st.markdown("<hr style='border-color:rgba(148,163,184,0.2); margin:24px 0;'>", unsafe_allow_html=True)
                    st.markdown(f"#### 🤖 AI Agronomist Chat")
                    st.markdown("<p style='font-size:0.85rem; color:#64748b;'>Ask follow-up questions about this crop.</p>", unsafe_allow_html=True)
                    
                    chat_container = st.container()
                    
                    # Display existing history
                    with chat_container:
                        for msg in st.session_state.chat_history:
                            with st.chat_message(msg["role"]):
                                st.write(msg["content"])
                                
                    # Get user input
                    user_msg = st.chat_input(f"Ask about {display_name} treatment...")
                    if user_msg:
                        with chat_container:
                            with st.chat_message("user"):
                                st.write(user_msg)
                            st.session_state.chat_history.append({"role": "user", "content": user_msg})
                            
                            with st.chat_message("assistant"):
                                with st.spinner("Thinking..."):
                                    agronomist_reply = get_agronomist_response(display_name, info, user_msg, st.session_state.chat_history[:-1])
                                    st.write(agronomist_reply)
                            st.session_state.chat_history.append({"role": "assistant", "content": agronomist_reply})

                except Exception as render_err:
                    st.error(f"⚠️ Error rendering results: {render_err}")
        else:"""

content = content.replace(old_pdf_block, new_pdf_block)

# 4. Remove original save_scan block since we pulled it up before rendering
old_save_block_1 = """                    display_name = name.replace("_", " ")

                    # Auto-save scan to database
                    save_scan(
                        user_id=st.session_state.user["id"],
                        disease_name=display_name,
                        confidence=round(conf, 2),
                        severity=severity,
                        latitude=map_lat if share_location else None,
                        longitude=map_lon if share_location else None
                    )"""

val = content.find(old_save_block_1)
if val != -1:
    content = content[:val] + """                    display_name = name.replace("_", " ")""" + content[val+len(old_save_block_1):]
else:
    print("WARNING: Could not find save_scan block!")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Patch applied successfully, length altered: {len(content)}")
