import streamlit as st

st.title("learning Streamlit Basics") 

st.header("This is a Header. Hi Header")

st.subheader("This is a subheader. Hi Subheader")

st.text("""Once a curious crow found a shiny pebble beside the river.
It carried the pebble home, believing it was a hidden treasure.
From that day on, the crow learned that curiosity can lead to wonderful discoveries.""" )

st.markdown("""### Python, **manish**
-Python is very boring
""")

st.success("you are successfully logged in!")

st.error("This is an error message!")

st.info("This is an information message!")

st.warning("This is a warning message!")

is_checked = st.checkbox("I agree to terms and conditions")
if is_checked:
    st.write("You have agreed to the terms and conditionns.")
else:
    st.write("You have not agreed to the terms and conditions.")


choosen_value = st.radio("select your gender",["Male","Female","Others"])
st.write("You hvae selected:",choosen_value)

st.selectbox("Select your country", ["India", "USA","Nepal","Canada"])
st.multiselect("Select your favourtie programming languages",["Python","C","Java","C++","Ruby"])

click_btn = st.button("Click me!")

if click_btn:
    st.write("Button clicked!")
    