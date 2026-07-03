"""
控制面板组件 - 视频控制、信息显示
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, 
    QLabel, QLineEdit, QFileDialog, QGroupBox, QGridLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QCheckBox, QToolButton, QColorDialog, QScrollArea, QFrame, QTabWidget,
    QSpinBox, QRadioButton, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QColor


class CollapsibleBox(QWidget):
    """可折叠容器 - 默认展开状态"""
    
    def __init__(self, title="", parent=None, expanded=True):
        super().__init__(parent)
        self._expanded = expanded
        
        self._btn = QToolButton(text=title)
        self._btn.setCheckable(True)
        self._btn.setChecked(expanded)
        self._btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._btn.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._btn.setStyleSheet("QToolButton { border: none; padding: 4px; font-weight: bold; }")
        self._btn.clicked.connect(self._toggle)
        
        self._content = QWidget()
        self._content.setVisible(expanded)
        self._inner_layout = QVBoxLayout(self._content)
        self._inner_layout.setContentsMargins(10, 5, 10, 5)
        
        lay = QVBoxLayout(self)
        lay.setSpacing(2)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._btn)
        lay.addWidget(self._content)
    
    def _toggle(self):
        checked = self._btn.isChecked()
        self._btn.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self._content.setVisible(checked)
    
    def innerLayout(self):
        return self._inner_layout
    
    def setExpanded(self, expanded: bool):
        """设置展开/折叠状态"""
        self._btn.setChecked(expanded)
        self._toggle()


class ControlPanel(QWidget):
    """控制面板"""
    
    # 信号
    load_video_clicked = pyqtSignal()
    load_trajectory_clicked = pyqtSignal()
    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    next_frame_clicked = pyqtSignal()
    prev_frame_clicked = pyqtSignal()
    slider_changed = pyqtSignal(int)
    goto_frame_clicked = pyqtSignal(int)
    save_results_clicked = pyqtSignal()
    analyze_save_clicked = pyqtSignal(str)
    analysis_view_clicked = pyqtSignal(str, object)
    analysis_next_clicked = pyqtSignal(str)
    analysis_view_should_clear = pyqtSignal()
    clear_data_clicked = pyqtSignal()
    show_points_changed = pyqtSignal(bool)
    show_trajectory_changed = pyqtSignal(bool)
    show_ids_changed = pyqtSignal(bool)
    show_footprints_changed = pyqtSignal(bool)
    trajectory_length_changed = pyqtSignal(int)
    add_mode_changed = pyqtSignal(bool)
    del_mode_changed = pyqtSignal(bool)
    add_footprint_mode_changed = pyqtSignal(bool)
    del_footprint_mode_changed = pyqtSignal(bool)
    end_mode_changed = pyqtSignal(bool)
    auto_sort_ids_changed = pyqtSignal(bool)
    
    # 取色器相关信号
    pick_color_mode_changed = pyqtSignal(bool)  # 取色模式切换
    
    correction_setup_clicked = pyqtSignal()
    correction_clear_clicked = pyqtSignal()
    correction_enabled_changed = pyqtSignal(bool)
    
    vertical_lines_enabled_changed = pyqtSignal(bool)
    vertical_lines_list_changed = pyqtSignal(object)
    vertical_lines_clear_clicked = pyqtSignal()
    track_note_changed = pyqtSignal(int, str)
    footprint_track_changed = pyqtSignal(int, object)
    keyframe_set_clicked = pyqtSignal(object)
    keyframe_target_text_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._new_point_color = (255, 0, 0)
        self._new_footprint_color = (0, 255, 255)
        self._end_point_color = (0, 255, 0)
        self._block_color = (177, 0, 50)  # 默认基准色
        self._updating_tracks_table = False
        self._updating_footprints_table = False
        self._setup_ui()
    
    def _setup_ui(self):
        # 主布局使用分页导航
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self.tab_widget = QTabWidget()
        self.tab_widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        main_layout.addWidget(self.tab_widget)
        
        page_video_scroll, page_video_layout = self._create_tab_page()
        page_track_scroll, page_track_layout = self._create_tab_page()
        page_analysis_scroll, page_analysis_layout = self._create_tab_page()
        self.tab_widget.addTab(page_video_scroll, "视频画面")
        self.tab_widget.addTab(page_track_scroll, "轨迹提取")
        self.tab_widget.addTab(page_analysis_scroll, "数据分析")
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # ===== 视频文件（包含视频信息）=====
        self.video_file_box = CollapsibleBox("视频文件", expanded=True)
        video_layout = QVBoxLayout()
        
        # 文件选择行
        file_row = QHBoxLayout()
        self.file_label = QLabel("未选择视频")
        self.file_label.setStyleSheet("color: gray;")
        self.load_btn = QPushButton("选择视频...")
        self.load_btn.clicked.connect(self.load_video_clicked.emit)
        file_row.addWidget(self.file_label, stretch=1)
        file_row.addWidget(self.load_btn)
        video_layout.addLayout(file_row)
        
        # 视频信息网格
        info_grid = QGridLayout()
        info_grid.setSpacing(5)
        
        info_grid.addWidget(QLabel("总帧数:"), 0, 0)
        self.total_frames_label = QLabel("-")
        info_grid.addWidget(self.total_frames_label, 0, 1)
        
        info_grid.addWidget(QLabel("总时长:"), 0, 2)
        self.total_time_label = QLabel("-")
        info_grid.addWidget(self.total_time_label, 0, 3)
        
        info_grid.addWidget(QLabel("FPS:"), 1, 0)
        self.fps_label = QLabel("-")
        info_grid.addWidget(self.fps_label, 1, 1)
        
        info_grid.addWidget(QLabel("分辨率:"), 1, 2)
        self.resolution_label = QLabel("-")
        info_grid.addWidget(self.resolution_label, 1, 3)
        
        video_layout.addLayout(info_grid)
        self.video_file_box.innerLayout().addLayout(video_layout)
        page_video_layout.addWidget(self.video_file_box)
        
        # ===== 播放控制 =====
        self.control_box = CollapsibleBox("播放控制", expanded=True)
        control_layout = QVBoxLayout()
        
        # 进度条
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.slider_changed.emit)
        # 禁用滑块的方向键控制，让全局键盘事件处理
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        control_layout.addWidget(self.slider)
        
        # 当前帧信息
        frame_info_layout = QHBoxLayout()
        frame_info_layout.addWidget(QLabel("当前帧:"))
        self.current_frame_label = QLabel("0")
        frame_info_layout.addWidget(self.current_frame_label)
        frame_info_layout.addStretch()
        
        # 跳转到指定帧
        frame_info_layout.addWidget(QLabel("跳转至帧:"))
        self.goto_frame_input = QLineEdit()
        self.goto_frame_input.setFixedWidth(60)
        self.goto_frame_input.setPlaceholderText("0")
        self.goto_frame_input.returnPressed.connect(self._on_goto_frame)
        frame_info_layout.addWidget(self.goto_frame_input)
        
        self.goto_btn = QPushButton("跳转")
        self.goto_btn.clicked.connect(self._on_goto_frame)
        frame_info_layout.addWidget(self.goto_btn)
        control_layout.addLayout(frame_info_layout)
        
        # 播放按钮
        btn_layout = QHBoxLayout()

        self.play_btn = QPushButton("▶播放")
        self.play_btn.clicked.connect(self._on_play)
        btn_layout.addWidget(self.play_btn)
        
        self.pause_btn = QPushButton("⏸暂停")
        self.pause_btn.clicked.connect(self._on_pause)
        self.pause_btn.setEnabled(False)
        btn_layout.addWidget(self.pause_btn)
        
        self.prev_btn = QPushButton("⏮上一帧")
        self.prev_btn.clicked.connect(self.prev_frame_clicked.emit)
        btn_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("下一帧⏭")
        self.next_btn.clicked.connect(self.next_frame_clicked.emit)
        btn_layout.addWidget(self.next_btn)
        control_layout.addLayout(btn_layout)
        
        self.control_box.innerLayout().addLayout(control_layout)
        page_track_layout.addWidget(self.control_box)
        
        # ===== 显示选项 =====
        self.display_box = CollapsibleBox("显示选项", expanded=True)
        display_layout = QVBoxLayout()
        display_layout.setSpacing(8)
        display_row1 = QHBoxLayout()
        display_row2 = QHBoxLayout()
        
        self.show_points_checkbox = QCheckBox("显示追踪点")
        self.show_points_checkbox.setChecked(True)
        self.show_points_checkbox.stateChanged.connect(self._on_show_points_changed)
        display_row1.addWidget(self.show_points_checkbox)
        
        self.show_trajectory_checkbox = QCheckBox("显示轨迹")
        self.show_trajectory_checkbox.setChecked(True)
        self.show_trajectory_checkbox.stateChanged.connect(self._on_show_trajectory_changed)
        display_row1.addWidget(self.show_trajectory_checkbox)
        
        self.show_ids_checkbox = QCheckBox("显示编号")
        self.show_ids_checkbox.setChecked(True)
        self.show_ids_checkbox.stateChanged.connect(self._on_show_ids_changed)
        display_row1.addWidget(self.show_ids_checkbox)
        
        self.show_footprints_checkbox = QCheckBox("显示脚印")
        self.show_footprints_checkbox.setChecked(True)
        self.show_footprints_checkbox.stateChanged.connect(self._on_show_footprints_changed)
        display_row1.addWidget(self.show_footprints_checkbox)
        display_row1.addStretch()
        
        display_row2.addWidget(QLabel("轨迹长度(帧):"))
        self.trajectory_length_spinbox = QSpinBox()
        self.trajectory_length_spinbox.setRange(1, 10000)
        self.trajectory_length_spinbox.setValue(10)
        self.trajectory_length_spinbox.valueChanged.connect(self.trajectory_length_changed.emit)
        display_row2.addWidget(self.trajectory_length_spinbox)
        display_row2.addStretch()
        
        display_layout.addLayout(display_row1)
        display_layout.addLayout(display_row2)
        self.display_box.innerLayout().addLayout(display_layout)
        page_video_layout.addWidget(self.display_box)
        
        self.correction_box = CollapsibleBox("画面校正与标记", expanded=True)
        correction_layout = QVBoxLayout()
        correction_layout.setSpacing(8)
        
        row_corr = QHBoxLayout()
        row_corr.setSpacing(10)
        
        self.setup_correction_btn = QPushButton("设置校正四点...")
        self.setup_correction_btn.clicked.connect(self.correction_setup_clicked.emit)
        row_corr.addWidget(self.setup_correction_btn)
        
        self.enable_correction_checkbox = QCheckBox("启用画面校正")
        self.enable_correction_checkbox.setChecked(False)
        self.enable_correction_checkbox.setEnabled(False)
        self.enable_correction_checkbox.stateChanged.connect(self._on_correction_enabled_changed)
        row_corr.addWidget(self.enable_correction_checkbox)
        
        self.clear_correction_btn = QPushButton("清除校正")
        self.clear_correction_btn.clicked.connect(self.correction_clear_clicked.emit)
        self.clear_correction_btn.setEnabled(False)
        row_corr.addWidget(self.clear_correction_btn)
        
        row_corr.addStretch()
        correction_layout.addLayout(row_corr)
        
        row_lines = QHBoxLayout()
        row_lines.setSpacing(10)
        
        self.show_vertical_lines_checkbox = QCheckBox("显示竖线标记")
        self.show_vertical_lines_checkbox.setChecked(True)
        self.show_vertical_lines_checkbox.stateChanged.connect(self._on_vertical_lines_enabled_changed)
        row_lines.addWidget(self.show_vertical_lines_checkbox)
        
        row_lines.addWidget(QLabel("X列表(px):"))
        self.vertical_lines_input = QLineEdit()
        self.vertical_lines_input.setPlaceholderText("例如：120, 360, 920")
        self.vertical_lines_input.editingFinished.connect(self._on_vertical_lines_list_changed)
        row_lines.addWidget(self.vertical_lines_input, stretch=1)
        
        self.clear_vertical_lines_btn = QPushButton("清空竖线")
        self.clear_vertical_lines_btn.clicked.connect(self.vertical_lines_clear_clicked.emit)
        row_lines.addWidget(self.clear_vertical_lines_btn)
        
        correction_layout.addLayout(row_lines)
        
        hint = QLabel("Shift+左键添加竖线，Shift+右键删除最近竖线")
        hint.setStyleSheet("color: #666;")
        correction_layout.addWidget(hint)
        
        self.correction_box.innerLayout().addLayout(correction_layout)
        page_video_layout.addWidget(self.correction_box)
        
        # ===== 数据读写 =====
        self.trajectory_box = CollapsibleBox("数据读写", expanded=True)
        trajectory_layout = QGridLayout()
        
        self.load_trajectory_btn = QPushButton("📂载入轨迹与画面校正")
        self.load_trajectory_btn.clicked.connect(self.load_trajectory_clicked.emit)
        trajectory_layout.addWidget(self.load_trajectory_btn, 0, 0)
        
        self.save_btn = QPushButton("💾保存结果")
        self.save_btn.clicked.connect(self.save_results_clicked.emit)
        trajectory_layout.addWidget(self.save_btn, 0, 1)

        self.clear_data_btn = QPushButton("清空所有轨迹和脚印")
        self.clear_data_btn.clicked.connect(self.clear_data_clicked.emit)
        trajectory_layout.addWidget(self.clear_data_btn, 1, 0, 1, 2)
        
        self.trajectory_box.innerLayout().addLayout(trajectory_layout)
        page_video_layout.addWidget(self.trajectory_box)
        
        # ===== 追踪点修改 =====
        self.edit_box = CollapsibleBox("追踪点修改", expanded=True)
        edit_layout = QVBoxLayout()
        edit_layout.setSpacing(8)
        
        # 第一行
        row1 = QHBoxLayout()
        row1.setSpacing(10)

        self.enable_auto_tracking_checkbox = QCheckBox("开启自动跟踪")
        self.enable_auto_tracking_checkbox.setChecked(False)
        self.enable_auto_tracking_checkbox.stateChanged.connect(self._on_enable_auto_tracking_changed)
        row1.addWidget(self.enable_auto_tracking_checkbox)
        
        self.add_point_checkbox = QCheckBox("增加追踪点（Alt+左键）")
        self.add_point_checkbox.setChecked(False)
        self.add_point_checkbox.stateChanged.connect(self._on_add_mode_changed)
        row1.addWidget(self.add_point_checkbox)
        
        # 颜色方块按钮
        self.color_preview = QPushButton()
        self.color_preview.setFixedSize(18, 18)
        self.color_preview.setCursor(Qt.CursorShape.PointingHandCursor)
        self.color_preview.clicked.connect(self._on_select_color)
        self._update_color_preview(self._new_point_color)
        row1.addWidget(self.color_preview)
        
        row1.addSpacing(20)
        
        self.del_point_checkbox = QCheckBox("删除追踪点（选中+Delete）")
        self.del_point_checkbox.setChecked(False)
        self.del_point_checkbox.stateChanged.connect(self._on_del_mode_changed)
        row1.addWidget(self.del_point_checkbox)
        row1.addStretch()
        edit_layout.addLayout(row1)
        
        # 第二行
        row2 = QHBoxLayout()
        row2.setSpacing(10)
        
        self.add_footprint_checkbox = QCheckBox("增加脚印点（左键）")
        self.add_footprint_checkbox.setChecked(True)
        self.add_footprint_checkbox.stateChanged.connect(self._on_add_footprint_mode_changed)
        row2.addWidget(self.add_footprint_checkbox)
        
        self.footprint_color_preview = QPushButton()
        self.footprint_color_preview.setFixedSize(18, 18)
        self.footprint_color_preview.setCursor(Qt.CursorShape.PointingHandCursor)
        self.footprint_color_preview.clicked.connect(self._on_select_footprint_color)
        self._update_footprint_color_preview(self._new_footprint_color)
        row2.addWidget(self.footprint_color_preview)
        
        self.del_footprint_checkbox = QCheckBox("删除脚印点（选中+Delete）")
        self.del_footprint_checkbox.setChecked(True)
        self.del_footprint_checkbox.stateChanged.connect(self._on_del_footprint_mode_changed)
        row2.addWidget(self.del_footprint_checkbox)
        
        row2.addStretch()
        edit_layout.addLayout(row2)
        
        # 第三行
        row3 = QHBoxLayout()
        row3.setSpacing(10)
        
        self.end_point_checkbox = QCheckBox("结束点（Ctrl+左键）")
        self.end_point_checkbox.setChecked(False)
        self.end_point_checkbox.stateChanged.connect(self._on_end_mode_changed)
        row3.addWidget(self.end_point_checkbox)
        
        # 结束点颜色方块
        self.end_color_preview = QPushButton()
        self.end_color_preview.setFixedSize(18, 18)
        self.end_color_preview.setCursor(Qt.CursorShape.PointingHandCursor)
        self.end_color_preview.clicked.connect(self._on_select_end_color)
        self._update_end_color_preview(self._end_point_color)
        row3.addWidget(self.end_color_preview)
        
        row3.addStretch()
        
        # 编号自动排序选项
        self.auto_sort_checkbox = QCheckBox("编号自动排序")
        self.auto_sort_checkbox.setChecked(True)
        self.auto_sort_checkbox.stateChanged.connect(self._on_auto_sort_changed)
        row3.addWidget(self.auto_sort_checkbox)
        edit_layout.addLayout(row3)

        self._sync_auto_tracking_controls(False)

        row4 = QHBoxLayout()
        row4.setSpacing(10)

        self.set_keyframe_btn = QPushButton("设为关键帧")
        self.set_keyframe_btn.clicked.connect(self._on_set_keyframe_clicked)
        row4.addWidget(self.set_keyframe_btn)

        self.keyframe_target_input = QLineEdit()
        self.keyframe_target_input.setPlaceholderText("追踪点ID，如：3 或 1,2")
        self.keyframe_target_input.textChanged.connect(self.keyframe_target_text_changed.emit)
        row4.addWidget(self.keyframe_target_input, stretch=1)

        self.keyframe_count_label = QLabel("此点已有0个关键帧")
        row4.addWidget(self.keyframe_count_label)
        row4.addStretch()
        edit_layout.addLayout(row4)
        
        self.edit_box.innerLayout().addLayout(edit_layout)
        page_track_layout.addWidget(self.edit_box)
        
        # ===== 色块自动追踪 =====
        self.color_track_box = CollapsibleBox("色块自动追踪", expanded=True)
        color_track_layout = QVBoxLayout()
        color_track_layout.setSpacing(8)
        
        # 第一行：色块设定
        color_setting_row = QHBoxLayout()
        color_setting_row.setSpacing(10)
        
        color_setting_row.addWidget(QLabel("基准颜色:"))
        
        # 基准颜色选择按钮
        self.block_color_btn = QPushButton()
        self.block_color_btn.setFixedSize(24, 24)
        self.block_color_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.block_color_btn.clicked.connect(self._on_select_block_color)
        self._update_block_color_preview()
        color_setting_row.addWidget(self.block_color_btn)
        
        # 取色器按钮
        self.pick_color_btn = QPushButton("🖱️")
        self.pick_color_btn.setFixedSize(24, 24)
        self.pick_color_btn.setToolTip("从视频画面取色")
        self.pick_color_btn.setCheckable(True)
        self.pick_color_btn.clicked.connect(self._on_pick_color_mode)
        color_setting_row.addWidget(self.pick_color_btn)
        
        color_setting_row.addSpacing(15)
        
        # 颜色容差
        color_setting_row.addWidget(QLabel("容差:"))
        self.color_tolerance_input = QLineEdit()
        self.color_tolerance_input.setFixedWidth(50)
        self.color_tolerance_input.setText("50")
        color_setting_row.addWidget(self.color_tolerance_input)
        
        color_setting_row.addSpacing(15)
        
        # 像素尺寸
        color_setting_row.addWidget(QLabel("像素尺寸:"))
        self.min_pixel_size_input = QLineEdit()
        self.min_pixel_size_input.setFixedWidth(50)
        self.min_pixel_size_input.setText("10")
        color_setting_row.addWidget(self.min_pixel_size_input)
        
        color_setting_row.addStretch()
        color_track_layout.addLayout(color_setting_row)
        
        # 第二行：功能复选框
        func_row = QHBoxLayout()
        func_row.setSpacing(15)
        
        self.auto_add_checkbox = QCheckBox("追踪点自动添加")
        self.auto_add_checkbox.setChecked(False)
        func_row.addWidget(self.auto_add_checkbox)
        
        self.auto_correct_checkbox = QCheckBox("追踪点自动校正")
        self.auto_correct_checkbox.setChecked(True)
        func_row.addWidget(self.auto_correct_checkbox)
        
        self.auto_end_checkbox = QCheckBox("追踪点自动结束")
        self.auto_end_checkbox.setChecked(False)
        func_row.addWidget(self.auto_end_checkbox)
        
        func_row.addStretch()
        color_track_layout.addLayout(func_row)
        
        self.color_track_box.innerLayout().addLayout(color_track_layout)
        page_track_layout.addWidget(self.color_track_box)
        
        # ===== 追踪点信息 =====
        self.tracks_box = CollapsibleBox("追踪点信息", expanded=True)
        tracks_layout = QVBoxLayout()
        
        self.track_stats_label = QLabel("追踪点总数：0 个，当前画面追踪点个数：0 个")
        tracks_layout.addWidget(self.track_stats_label)
        
        self.disappearing_label = QLabel("正在结束的追踪点：无")
        tracks_layout.addWidget(self.disappearing_label)
        
        self.tracks_table = QTableWidget()
        self.tracks_table.setColumnCount(8)
        self.tracks_table.setHorizontalHeaderLabels(["ID", "备注", "颜色", "起始帧", "关键帧", "X", "Y", "色块"])
        
        # 设置列宽模式
        header = self.tracks_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)  # ID
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)  # 备注
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)  # 颜色
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)  # 起始帧
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)  # 关键帧
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)  # X
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)  # Y
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Fixed)  # 在色块上
        
        # 设置固定列宽
        self.tracks_table.setColumnWidth(0, 40)   # ID
        self.tracks_table.setColumnWidth(1, 70)   # 备注
        self.tracks_table.setColumnWidth(2, 50)   # 颜色
        self.tracks_table.setColumnWidth(3, 60)   # 起始帧
        self.tracks_table.setColumnWidth(4, 55)   # 关键帧
        self.tracks_table.setColumnWidth(7, 70)   # 在色块上
        
        self.tracks_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tracks_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked |
            QAbstractItemView.EditTrigger.EditKeyPressed |
            QAbstractItemView.EditTrigger.SelectedClicked
        )
        self.tracks_table.verticalHeader().setDefaultSectionSize(18)
        self.tracks_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.tracks_table.itemChanged.connect(self._on_tracks_table_item_changed)
        
        # 设置表格滚动条
        self.tracks_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tracks_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # 设置表格最大高度，避免占用过多空间
        self.tracks_table.setMaximumHeight(200)
        
        tracks_layout.addWidget(self.tracks_table)
        
        self.footprint_stats_label = QLabel("当前显示脚印点：0 个")
        tracks_layout.addWidget(self.footprint_stats_label)
        
        self.footprints_table = QTableWidget()
        self.footprints_table.setColumnCount(6)
        self.footprints_table.setHorizontalHeaderLabels(["脚印ID", "对应追踪点", "颜色", "起始帧", "X", "Y"])
        footprint_header = self.footprints_table.horizontalHeader()
        footprint_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        footprint_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        footprint_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        footprint_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        footprint_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        footprint_header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.footprints_table.setColumnWidth(0, 40)
        self.footprints_table.setColumnWidth(1, 70)
        self.footprints_table.setColumnWidth(2, 50)
        self.footprints_table.setColumnWidth(3, 60)
        self.footprints_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.footprints_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked |
            QAbstractItemView.EditTrigger.EditKeyPressed |
            QAbstractItemView.EditTrigger.SelectedClicked
        )
        self.footprints_table.verticalHeader().setDefaultSectionSize(18)
        self.footprints_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.footprints_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.footprints_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.footprints_table.setMaximumHeight(120)
        self.footprints_table.itemChanged.connect(self._on_footprints_table_item_changed)
        tracks_layout.addWidget(self.footprints_table)
        
        self.tracks_box.innerLayout().addLayout(tracks_layout)
        page_track_layout.addWidget(self.tracks_box)
        
        mode_row = QHBoxLayout()
        mode_row.setSpacing(12)
        mode_row.addWidget(QLabel("场景模式:"))

        self.single_scene_radio = QRadioButton("单人场景")
        self.single_scene_radio.setChecked(True)
        self.single_scene_radio.toggled.connect(self._on_analysis_scene_mode_toggled)
        mode_row.addWidget(self.single_scene_radio)

        self.queue_scene_radio = QRadioButton("队列场景")
        self.queue_scene_radio.toggled.connect(self._on_analysis_scene_mode_toggled)
        mode_row.addWidget(self.queue_scene_radio)

        self.crowd_scene_radio = QRadioButton("人群场景")
        self.crowd_scene_radio.toggled.connect(self._on_analysis_scene_mode_toggled)
        mode_row.addWidget(self.crowd_scene_radio)
        mode_row.addStretch()
        page_analysis_layout.addLayout(mode_row)

        # ===== 数据处理 =====
        self.trajectory_box = CollapsibleBox("数据处理", expanded=True)
        trajectory_layout = QGridLayout()

        process_hint = QLabel("当前分析时仅保留中间 5m 区间内的数据；人群场景暂时只保存原始结果，不生成 Analyze 文件。")
        process_hint.setWordWrap(True)
        process_hint.setStyleSheet("color: #666;")
        trajectory_layout.addWidget(process_hint, 0, 0, 1, 2)

        self.analyze_save_btn = QPushButton("处理数据并保存文件")
        self.analyze_save_btn.clicked.connect(self._on_analyze_save_clicked)
        trajectory_layout.addWidget(self.analyze_save_btn, 1, 0, 1, 2)
        
        self.trajectory_box.innerLayout().addLayout(trajectory_layout)
        page_analysis_layout.addWidget(self.trajectory_box)

        self.analysis_box = CollapsibleBox("结果画面显示", expanded=True)
        analysis_layout = QVBoxLayout()
        analysis_layout.setSpacing(8)

        self.analysis_view_hint_label = QLabel()
        self.analysis_view_hint_label.setWordWrap(True)
        self.analysis_view_hint_label.setStyleSheet("color: #666;")
        analysis_layout.addWidget(self.analysis_view_hint_label)

        analysis_query_row = QHBoxLayout()
        analysis_query_row.addWidget(QLabel("查看 track_id:"))
        self.analysis_track_id_input = QLineEdit()
        self.analysis_track_id_input.setPlaceholderText("输入轨迹 ID")
        self.analysis_track_id_input.setMaxLength(6)
        self.analysis_track_id_input.setFixedWidth(80)
        self.analysis_track_id_input.returnPressed.connect(self._on_analysis_view_clicked)
        analysis_query_row.addWidget(self.analysis_track_id_input)
        self.analysis_view_btn = QPushButton("查看结果")
        self.analysis_view_btn.clicked.connect(self._on_analysis_view_clicked)
        analysis_query_row.addWidget(self.analysis_view_btn)
        self.analysis_next_btn = QPushButton("查看下一个")
        self.analysis_next_btn.clicked.connect(self._on_analysis_next_clicked)
        analysis_query_row.addWidget(self.analysis_next_btn)
        analysis_layout.addLayout(analysis_query_row)

        self.analysis_status_label = QLabel("未选择 track_id")
        analysis_layout.addWidget(self.analysis_status_label)

        stats_grid = QGridLayout()
        stats_grid.setSpacing(6)
        self.analysis_track_id_value = QLabel("-")
        self.analysis_note_value = QLabel("-")
        self.analysis_start_frame_value = QLabel("-")
        self.analysis_point_count_value = QLabel("-")
        self.analysis_mean_speed_value = QLabel("-")
        self.analysis_speed_variance_value = QLabel("-")
        self.analysis_speed_range_value = QLabel("-")
        self.analysis_footprint_count_value = QLabel("-")
        self.analysis_mean_stride_value = QLabel("-")
        self.analysis_mean_interval_value = QLabel("-")
        self.analysis_mean_peak_valley_value = QLabel("-")
        self.analysis_mean_peak_height_value = QLabel("-")

        stats_items = [
            ("track_id:", self.analysis_track_id_value, 0, 0),
            ("note:", self.analysis_note_value, 0, 2),
            ("起始帧:", self.analysis_start_frame_value, 1, 0),
            ("轨迹点数:", self.analysis_point_count_value, 1, 2),
            ("平均速度(m/s):", self.analysis_mean_speed_value, 2, 0),
            ("速度方差:", self.analysis_speed_variance_value, 2, 2),
            ("速度范围(m/s):", self.analysis_speed_range_value, 3, 0),
            ("脚印点数:", self.analysis_footprint_count_value, 3, 2),
            ("平均步幅(m):", self.analysis_mean_stride_value, 4, 0),
            ("平均间隔(s):", self.analysis_mean_interval_value, 4, 2),
            ("平均峰谷差(m):", self.analysis_mean_peak_valley_value, 5, 0),
            ("平均峰值高度(m):", self.analysis_mean_peak_height_value, 5, 2),
        ]
        for text, widget, row, col in stats_items:
            stats_grid.addWidget(QLabel(text), row, col)
            stats_grid.addWidget(widget, row, col + 1)
        analysis_layout.addLayout(stats_grid)

        analysis_layout.addWidget(QLabel("5m 区间轨迹点:"))
        self.analysis_track_points_text = QTextEdit()
        self.analysis_track_points_text.setReadOnly(True)
        self.analysis_track_points_text.setMaximumHeight(110)
        analysis_layout.addWidget(self.analysis_track_points_text)

        analysis_layout.addWidget(QLabel("5m 区间脚印点:"))
        self.analysis_footprints_text = QTextEdit()
        self.analysis_footprints_text.setReadOnly(True)
        self.analysis_footprints_text.setMaximumHeight(110)
        analysis_layout.addWidget(self.analysis_footprints_text)
        self.analysis_box.innerLayout().addLayout(analysis_layout)
        page_analysis_layout.addWidget(self.analysis_box)
        self._update_analysis_scene_hint()
        self.clear_analysis_result_summary()
        
        page_video_layout.addStretch()
        page_track_layout.addStretch()
        page_analysis_layout.addStretch()

    def _create_tab_page(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        scroll.setWidget(container)
        return scroll, layout
    
    def _on_play(self):
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.play_clicked.emit()
    
    def _on_pause(self):
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_clicked.emit()
    
    def _on_goto_frame(self):
        try:
            frame_idx = int(self.goto_frame_input.text())
            self.goto_frame_clicked.emit(frame_idx)
        except ValueError:
            pass
    
    def _on_show_points_changed(self, state):
        self.show_points_changed.emit(state == Qt.CheckState.Checked.value)
    
    def _on_show_trajectory_changed(self, state):
        self.show_trajectory_changed.emit(state == Qt.CheckState.Checked.value)
    
    def _on_show_ids_changed(self, state):
        self.show_ids_changed.emit(state == Qt.CheckState.Checked.value)

    def _on_show_footprints_changed(self, state):
        self.show_footprints_changed.emit(state == Qt.CheckState.Checked.value)
    
    def _on_add_mode_changed(self, state):
        self.add_mode_changed.emit(state == Qt.CheckState.Checked.value)
    
    def _on_del_mode_changed(self, state):
        self.del_mode_changed.emit(state == Qt.CheckState.Checked.value)
    
    def _on_add_footprint_mode_changed(self, state):
        self.add_footprint_mode_changed.emit(state == Qt.CheckState.Checked.value)
    
    def _on_del_footprint_mode_changed(self, state):
        self.del_footprint_mode_changed.emit(state == Qt.CheckState.Checked.value)
    
    def _on_end_mode_changed(self, state):
        self.end_mode_changed.emit(state == Qt.CheckState.Checked.value)

    def _on_enable_auto_tracking_changed(self, state):
        enabled = state == Qt.CheckState.Checked.value
        self._sync_auto_tracking_controls(enabled)

    def _sync_auto_tracking_controls(self, enabled: bool):
        for checkbox in (self.add_point_checkbox, self.del_point_checkbox, self.end_point_checkbox):
            checkbox.blockSignals(True)
            checkbox.setChecked(enabled)
            checkbox.blockSignals(False)
            checkbox.setEnabled(enabled)

        self.add_mode_changed.emit(enabled)
        self.del_mode_changed.emit(enabled)
        self.end_mode_changed.emit(enabled)
    
    def _on_auto_sort_changed(self, state):
        self.auto_sort_ids_changed.emit(state == Qt.CheckState.Checked.value)
    
    def _on_select_color(self):
        color = QColorDialog.getColor(QColor(*self._new_point_color), self, "选择新点颜色")
        if color.isValid():
            self._new_point_color = (color.red(), color.green(), color.blue())
            self._update_color_preview(self._new_point_color)
    
    def _on_select_footprint_color(self):
        color = QColorDialog.getColor(QColor(*self._new_footprint_color), self, "选择脚印点颜色")
        if color.isValid():
            self._new_footprint_color = (color.red(), color.green(), color.blue())
            self._update_footprint_color_preview(self._new_footprint_color)
    
    def _update_color_preview(self, rgb):
        self.color_preview.setStyleSheet(
            f"QPushButton {{ background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); "
            f"border: 1px solid gray; border-radius: 2px; }}"
        )
    
    def _update_footprint_color_preview(self, rgb):
        self.footprint_color_preview.setStyleSheet(
            f"QPushButton {{ background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); "
            f"border: 1px solid gray; border-radius: 2px; }}"
        )
    
    def _on_select_end_color(self):
        color = QColorDialog.getColor(QColor(*self._end_point_color), self, "选择结束点颜色")
        if color.isValid():
            self._end_point_color = (color.red(), color.green(), color.blue())
            self._update_end_color_preview(self._end_point_color)
    
    def _update_end_color_preview(self, rgb):
        self.end_color_preview.setStyleSheet(
            f"QPushButton {{ background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); "
            f"border: 1px solid gray; border-radius: 2px; }}"
        )
    
    def _on_select_block_color(self):
        color = QColorDialog.getColor(QColor(*self._block_color), self, "选择色块基准颜色")
        if color.isValid():
            self._block_color = (color.red(), color.green(), color.blue())
            self._update_block_color_preview()
    
    def _on_pick_color_mode(self):
        """取色器模式切换"""
        is_active = self.pick_color_btn.isChecked()
        self.pick_color_mode_changed.emit(is_active)
        if is_active:
            self.pick_color_btn.setStyleSheet("background-color: lightblue;")
        else:
            self.pick_color_btn.setStyleSheet("")
    
    def set_pick_color_mode(self, active: bool):
        """设置取色器模式状态（外部调用）"""
        self.pick_color_btn.setChecked(active)
        if active:
            self.pick_color_btn.setStyleSheet("background-color: lightblue;")
        else:
            self.pick_color_btn.setStyleSheet("")
    
    def _update_block_color_preview(self):
        r, g, b = self._block_color
        self.block_color_btn.setStyleSheet(
            f"QPushButton {{ background-color: rgb({r}, {g}, {b}); "
            f"border: 2px solid gray; border-radius: 3px; }}"
        )
    
    def set_block_color(self, color: tuple):
        """设置色块基准颜色（从取色器获取）"""
        self._block_color = color
        self._update_block_color_preview()
    
    def get_new_point_color(self):
        return self._new_point_color
    
    def get_new_footprint_color(self):
        return self._new_footprint_color
    
    def get_end_point_color(self):
        return self._end_point_color
    
    def is_auto_sort_enabled(self):
        return self.auto_sort_checkbox.isChecked()
    
    def get_block_color(self) -> tuple:
        return self._block_color
    
    def get_color_tolerance(self) -> int:
        try:
            return int(self.color_tolerance_input.text())
        except ValueError:
            return 30
    
    def get_min_pixel_size(self) -> int:
        try:
            return int(self.min_pixel_size_input.text())
        except ValueError:
            return 20
    
    def is_auto_add_enabled(self) -> bool:
        return self.is_auto_tracking_enabled() and self.auto_add_checkbox.isChecked()
    
    def is_auto_correct_enabled(self) -> bool:
        return self.is_auto_tracking_enabled() and self.auto_correct_checkbox.isChecked()
    
    def is_auto_end_enabled(self) -> bool:
        return self.is_auto_tracking_enabled() and self.auto_end_checkbox.isChecked()

    def is_auto_tracking_enabled(self) -> bool:
        return self.enable_auto_tracking_checkbox.isChecked()

    def is_add_mode_enabled(self) -> bool:
        return self.add_point_checkbox.isChecked() and self.add_point_checkbox.isEnabled()

    def is_del_mode_enabled(self) -> bool:
        return self.del_point_checkbox.isChecked() and self.del_point_checkbox.isEnabled()

    def is_end_mode_enabled(self) -> bool:
        return self.end_point_checkbox.isChecked() and self.end_point_checkbox.isEnabled()
    
    def is_pick_color_mode(self) -> bool:
        """是否处于取色模式"""
        return self.pick_color_btn.isChecked()
    
    def set_video_info(self, filepath: str, total_frames: int, fps: float, width: int, height: int):
        self.file_label.setText(filepath.split('/')[-1].split('\\')[-1])
        self.file_label.setStyleSheet("color: black;")
        
        self.total_frames_label.setText(str(total_frames))
        self.fps_label.setText(f"{fps:.2f}")
        self.resolution_label.setText(f"{width} x {height}")
        
        total_seconds = total_frames / fps if fps > 0 else 0
        self.total_time_label.setText(self._format_time(total_seconds))
        
        self.slider.setRange(0, total_frames - 1)
    
    def set_current_frame(self, frame_idx: int, fps: float):
        self.current_frame_label.setText(str(frame_idx))
        self.slider.blockSignals(True)
        self.slider.setValue(frame_idx)
        self.slider.blockSignals(False)
    
    def update_tracks_table(self, tracks_dict: dict, disappearing_tracks: dict = None, 
                           total_tracks: int = 0, on_block_status: dict = None, current_frame_idx: int = 0):
        """更新追踪点列表
        
        Args:
            tracks_dict: 追踪点字典
            disappearing_tracks: 正在消失的追踪点
            total_tracks: 追踪点总数
            on_block_status: 追踪点是否在色块上的状态 {track_id: (bool, color) 或 (bool, None)}
        """
        disappearing_tracks = disappearing_tracks or {}
        on_block_status = on_block_status or {}
        
        current_count = len(tracks_dict) + len(disappearing_tracks)
        self.track_stats_label.setText(f"追踪点总数：{total_tracks} 个，当前画面追踪点个数：{current_count} 个")
        
        if disappearing_tracks:
            ids_str = "，".join(str(tid) for tid in sorted(disappearing_tracks.keys()))
            self.disappearing_label.setText(f"正在结束的追踪点：{ids_str}")
        else:
            self.disappearing_label.setText("正在结束的追踪点：无")
        
        self._updating_tracks_table = True
        self.tracks_table.blockSignals(True)
        self.tracks_table.setRowCount(len(tracks_dict))
        
        for i, (track_id, track) in enumerate(sorted(tracks_dict.items())):
            if track.positions:
                _, pos = track.positions[-1]
                id_item = QTableWidgetItem(str(track_id))
                id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.tracks_table.setItem(i, 0, id_item)
                
                note_item = QTableWidgetItem(track.note)
                note_item.setData(Qt.ItemDataRole.UserRole, track_id)
                self.tracks_table.setItem(i, 1, note_item)
                
                # 颜色列（追踪点颜色）
                color_label = QLabel()
                color_label.setFixedSize(16, 16)
                r, g, b = track.color
                color_label.setStyleSheet(
                    f"background-color: rgb({r}, {g}, {b}); border: 1px solid gray;"
                )
                color_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.tracks_table.setCellWidget(i, 2, color_label)
                
                start_frame_item = QTableWidgetItem(str(track.start_frame))
                start_frame_item.setFlags(start_frame_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.tracks_table.setItem(i, 3, start_frame_item)

                keyframe_text = "是" if track.keyframe_frames.get(current_frame_idx, False) else ""
                keyframe_item = QTableWidgetItem(keyframe_text)
                keyframe_item.setFlags(keyframe_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.tracks_table.setItem(i, 4, keyframe_item)
                
                x_item = QTableWidgetItem(f"{pos[0]:.1f}")
                x_item.setFlags(x_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.tracks_table.setItem(i, 5, x_item)
                y_item = QTableWidgetItem(f"{pos[1]:.1f}")
                y_item.setFlags(y_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.tracks_table.setItem(i, 6, y_item)
                
                # 在色块上列
                # 先清除该单元格的任何Widget（防止之前的颜色方块残留）
                self.tracks_table.removeCellWidget(i, 7)
                
                # 检查是否开启了色块追踪功能
                color_tracking_enabled = self.is_auto_add_enabled() or self.is_auto_correct_enabled() or self.is_auto_end_enabled()
                
                if not color_tracking_enabled:
                    # 未开启色块功能，显示"无"
                    status_item = QTableWidgetItem("无")
                    status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.tracks_table.setItem(i, 7, status_item)
                else:
                    # 开启了色块功能
                    status_info = on_block_status.get(track_id, None)
                    
                    if status_info is None:
                        # 开启了功能但没有该追踪点的记录（可能是手动添加的）
                        status_item = QTableWidgetItem("无")
                        status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        self.tracks_table.setItem(i, 7, status_item)
                    elif isinstance(status_info, tuple):
                        is_on_block, block_color = status_info
                        if is_on_block and block_color:
                            # 在色块上，显示色块颜色方块
                            block_color_label = QLabel()
                            block_color_label.setFixedSize(16, 16)
                            r, g, b = block_color
                            block_color_label.setStyleSheet(
                                f"background-color: rgb({r}, {g}, {b}); border: 1px solid gray;"
                            )
                            block_color_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                            self.tracks_table.setCellWidget(i, 7, block_color_label)
                        else:
                            # 不在色块上
                            status_item = QTableWidgetItem("无")
                            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                            self.tracks_table.setItem(i, 7, status_item)
                    else:
                        # 兼容旧格式（bool）
                        if status_info:
                            status_item = QTableWidgetItem("")
                        else:
                            status_item = QTableWidgetItem("无")
                        status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        self.tracks_table.setItem(i, 7, status_item)
        
        self.tracks_table.blockSignals(False)
        self._updating_tracks_table = False
    
    def update_footprints_table(self, footprints: list):
        self.footprint_stats_label.setText(f"当前显示脚印点：{len(footprints)} 个")
        self._updating_footprints_table = True
        self.footprints_table.blockSignals(True)
        self.footprints_table.setRowCount(len(footprints))
        
        for i, footprint in enumerate(footprints):
            id_item = QTableWidgetItem(str(footprint["id"]))
            id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.footprints_table.setItem(i, 0, id_item)
            
            related_track = footprint.get("track_id")
            related_text = "" if related_track is None else str(related_track)
            related_item = QTableWidgetItem(related_text)
            related_item.setData(Qt.ItemDataRole.UserRole, footprint["id"])
            self.footprints_table.setItem(i, 1, related_item)
            
            color_label = QLabel()
            color_label.setFixedSize(16, 16)
            r, g, b = footprint["color"]
            color_label.setStyleSheet(
                f"background-color: rgb({r}, {g}, {b}); border: 1px solid gray;"
            )
            color_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.footprints_table.setCellWidget(i, 2, color_label)
            
            start_item = QTableWidgetItem(str(footprint["frame_idx"]))
            start_item.setFlags(start_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.footprints_table.setItem(i, 3, start_item)
            
            x_item = QTableWidgetItem(f'{footprint["x"]:.1f}')
            x_item.setFlags(x_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.footprints_table.setItem(i, 4, x_item)
            
            y_item = QTableWidgetItem(f'{footprint["y"]:.1f}')
            y_item.setFlags(y_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.footprints_table.setItem(i, 5, y_item)
        
        self.footprints_table.blockSignals(False)
        self._updating_footprints_table = False
    
    def _format_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def _on_correction_enabled_changed(self, state):
        self.correction_enabled_changed.emit(state == Qt.CheckState.Checked.value)
    
    def _on_vertical_lines_enabled_changed(self, state):
        self.vertical_lines_enabled_changed.emit(state == Qt.CheckState.Checked.value)
    
    def _on_vertical_lines_list_changed(self):
        raw = self.vertical_lines_input.text().strip()
        if not raw:
            self.vertical_lines_list_changed.emit([])
            return
        
        parts = [p.strip() for p in raw.replace("，", ",").split(",")]
        xs = []
        for p in parts:
            if not p:
                continue
            try:
                xs.append(float(p))
            except ValueError:
                continue
        
        self.vertical_lines_list_changed.emit(xs)
    
    def set_correction_available(self, available: bool, enabled: bool = False):
        self.enable_correction_checkbox.setEnabled(available)
        self.clear_correction_btn.setEnabled(available)
        self.enable_correction_checkbox.blockSignals(True)
        self.enable_correction_checkbox.setChecked(enabled if available else False)
        self.enable_correction_checkbox.blockSignals(False)
    
    def set_vertical_lines_positions(self, xs: list):
        text = ", ".join(str(int(x)) if float(x).is_integer() else f"{x:.1f}" for x in xs)
        self.vertical_lines_input.blockSignals(True)
        self.vertical_lines_input.setText(text)
        self.vertical_lines_input.blockSignals(False)

    def get_trajectory_length_frames(self) -> int:
        return self.trajectory_length_spinbox.value()

    def set_trajectory_length_frames(self, frame_count: int):
        self.trajectory_length_spinbox.blockSignals(True)
        self.trajectory_length_spinbox.setValue(max(1, int(frame_count)))
        self.trajectory_length_spinbox.blockSignals(False)

    def get_analysis_scene_mode(self) -> str:
        if self.queue_scene_radio.isChecked():
            return "queue"
        if self.crowd_scene_radio.isChecked():
            return "crowd"
        return "single"

    def _on_analysis_scene_mode_toggled(self, checked: bool):
        if not checked:
            return
        self._update_analysis_scene_hint()

    def _on_tab_changed(self, index: int):
        if index == 1:  # 轨迹提取
            self.analysis_view_should_clear.emit()

    def _on_analyze_save_clicked(self):
        self.analyze_save_clicked.emit(self.get_analysis_scene_mode())

    def _on_analysis_view_clicked(self):
        self.analysis_view_clicked.emit(self.get_analysis_scene_mode(), self.get_analysis_target_track_id())

    def _on_analysis_next_clicked(self):
        self.analysis_next_clicked.emit(self.get_analysis_scene_mode())

    def get_analysis_target_track_id(self):
        raw = self.analysis_track_id_input.text().strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def set_analysis_target_track_id(self, track_id):
        text = "" if track_id is None else str(int(track_id))
        self.analysis_track_id_input.blockSignals(True)
        self.analysis_track_id_input.setText(text)
        self.analysis_track_id_input.blockSignals(False)

    def _update_analysis_scene_hint(self):
        scene_mode = self.get_analysis_scene_mode()
        if scene_mode == "single":
            text = "单人模式下可指定 track_id，左侧显示该 ID 起始帧，并叠加 5m 区间内全部轨迹和脚印。"
        elif scene_mode == "queue":
            text = "队列模式下当前可先按单人方式查看指定 track_id；多人关联显示可后续扩展。"
        else:
            text = "人群模式下当前可先按单人方式查看指定 track_id；更多群体结果显示后续扩展。"
        self.analysis_view_hint_label.setText(text)

    def clear_analysis_result_summary(self, message: str = "未选择 track_id"):
        self.analysis_status_label.setText(message)
        for widget in (
            self.analysis_track_id_value,
            self.analysis_note_value,
            self.analysis_start_frame_value,
            self.analysis_point_count_value,
            self.analysis_mean_speed_value,
            self.analysis_speed_variance_value,
            self.analysis_speed_range_value,
            self.analysis_footprint_count_value,
            self.analysis_mean_stride_value,
            self.analysis_mean_interval_value,
            self.analysis_mean_peak_valley_value,
            self.analysis_mean_peak_height_value,
        ):
            widget.setText("-")
        self.analysis_track_points_text.setPlainText("")
        self.analysis_footprints_text.setPlainText("")

    def set_analysis_result_summary(self, summary: dict):
        self.analysis_status_label.setText(summary.get("status", ""))
        self.analysis_track_id_value.setText(str(summary.get("track_id", "-")))
        self.analysis_note_value.setText(str(summary.get("note", "-")))
        self.analysis_start_frame_value.setText(str(summary.get("start_frame", "-")))
        self.analysis_point_count_value.setText(str(summary.get("point_count", "-")))
        self.analysis_mean_speed_value.setText(str(summary.get("mean_speed", "-")))
        self.analysis_speed_variance_value.setText(str(summary.get("speed_variance", "-")))
        self.analysis_speed_range_value.setText(str(summary.get("speed_range", "-")))
        self.analysis_footprint_count_value.setText(str(summary.get("footprint_count", "-")))
        self.analysis_mean_stride_value.setText(str(summary.get("mean_stride", "-")))
        self.analysis_mean_interval_value.setText(str(summary.get("mean_interval", "-")))
        self.analysis_mean_peak_valley_value.setText(str(summary.get("mean_peak_valley", "-")))
        self.analysis_mean_peak_height_value.setText(str(summary.get("mean_peak_height", "-")))
        self.analysis_track_points_text.setPlainText(summary.get("track_points_text", ""))
        self.analysis_footprints_text.setPlainText(summary.get("footprints_text", ""))

    def get_navigation_step_frames(self) -> int:
        return self.navigation_step_spinbox.value()

    def _on_set_keyframe_clicked(self):
        self.keyframe_set_clicked.emit(self.get_keyframe_target_ids())

    def get_keyframe_target_ids(self) -> list:
        raw = self.keyframe_target_input.text().strip()
        if not raw:
            return []
        parts = [p.strip() for p in raw.replace("，", ",").split(",")]
        ids = []
        for part in parts:
            if not part:
                continue
            try:
                ids.append(int(part))
            except ValueError:
                continue
        return ids

    def set_keyframe_target_track_id(self, track_id):
        text = "" if track_id is None else str(int(track_id))
        self.keyframe_target_input.blockSignals(True)
        self.keyframe_target_input.setText(text)
        self.keyframe_target_input.blockSignals(False)

    def set_keyframe_count(self, count: int):
        self.keyframe_count_label.setText(f"此点已有{max(0, int(count))}个关键帧")

    def remap_keyframe_target_ids(self, id_map: dict):
        ids = self.get_keyframe_target_ids()
        if not ids:
            return
        remapped = [id_map.get(track_id, track_id) for track_id in ids]
        self.keyframe_target_input.blockSignals(True)
        self.keyframe_target_input.setText(",".join(str(track_id) for track_id in remapped))
        self.keyframe_target_input.blockSignals(False)

    def _on_tracks_table_item_changed(self, item: QTableWidgetItem):
        if self._updating_tracks_table or item.column() != 1:
            return
        
        track_id = item.data(Qt.ItemDataRole.UserRole)
        if track_id is None:
            id_item = self.tracks_table.item(item.row(), 0)
            if id_item is None:
                return
            try:
                track_id = int(id_item.text())
            except ValueError:
                return
        
        self.track_note_changed.emit(int(track_id), item.text())
    
    def _on_footprints_table_item_changed(self, item: QTableWidgetItem):
        if self._updating_footprints_table or item.column() != 1:
            return
        
        footprint_id = item.data(Qt.ItemDataRole.UserRole)
        if footprint_id is None:
            id_item = self.footprints_table.item(item.row(), 0)
            if id_item is None:
                return
            try:
                footprint_id = int(id_item.text())
            except ValueError:
                return
        
        text = item.text().strip()
        if not text:
            self.footprint_track_changed.emit(int(footprint_id), None)
            return
        
        try:
            self.footprint_track_changed.emit(int(footprint_id), int(text))
        except ValueError:
            self.footprint_track_changed.emit(int(footprint_id), None)
