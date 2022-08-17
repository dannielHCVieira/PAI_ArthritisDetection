import dearpygui.dearpygui as dpg

def print_me(sender):
    print(f"Menu Item: {sender}")

dpg.create_context()
dpg.create_viewport(title='AtroseID', width=800, height=800)

with dpg.viewport_menu_bar():
    with dpg.menu(label="File"):
        dpg.add_menu_item(label="Open image", callback=print_me)

    with dpg.menu(label="Tools"):
        dpg.add_menu_item(label="Cut/Select subimage", callback=print_me)
        dpg.add_menu_item(label="Search with subimage", callback=print_me)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()