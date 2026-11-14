# SPDX-License-Identifier: Apache-2.0
"""
Generate a Large-Scale Multi-Task Civic Dataset across all 63 provinces of Vietnam.
- 36,000 total samples (3,000 samples per category x 12 categories)
- Perfectly balanced 3 urgency levels: 12,000 High, 12,000 Normal, 12,000 Low
- 4 Core Departments: infrastructure, environment, transport, intake
- Authentic realistic GPS coordinates (lat, lon) per province
- ISO 8601 timestamps
- Rich linguistic variety and phrasing
"""

import json
import random
import csv
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

# 63 Provinces & Municipalities with approximate bounding boxes (lat_min, lat_max, lon_min, lon_max)
PROVINCES_GEO = {
    # Northern Vietnam
    "Hà Nội": (20.85, 21.25, 105.65, 105.95),
    "Hải Phòng": (20.75, 20.95, 106.60, 106.80),
    "Quảng Ninh": (20.80, 21.50, 106.80, 107.90),
    "Bắc Ninh": (21.05, 21.20, 106.00, 106.20),
    "Hải Dương": (20.80, 21.10, 106.20, 106.50),
    "Hưng Yên": (20.65, 20.95, 105.95, 106.15),
    "Hà Nam": (20.45, 20.65, 105.85, 106.05),
    "Nam Định": (20.15, 20.45, 106.10, 106.35),
    "Thái Bình": (20.35, 20.65, 106.25, 106.55),
    "Ninh Bình": (20.15, 20.35, 105.80, 106.05),
    "Vĩnh Phúc": (21.25, 21.45, 105.50, 105.75),
    "Phú Thọ": (21.20, 21.50, 105.10, 105.40),
    "Bắc Giang": (21.20, 21.45, 106.10, 106.50),
    "Thái Nguyên": (21.45, 21.75, 105.70, 106.00),
    "Lạng Sơn": (21.70, 22.10, 106.60, 107.00),
    "Cao Bằng": (22.50, 22.85, 106.10, 106.45),
    "Bắc Kạn": (22.05, 22.35, 105.75, 106.05),
    "Tuyên Quang": (21.70, 22.10, 105.15, 105.35),
    "Hà Giang": (22.70, 23.10, 104.90, 105.20),
    "Yên Bái": (21.65, 21.90, 104.80, 105.00),
    "Lào Cai": (22.40, 22.60, 103.90, 104.10),
    "Sơn La": (21.25, 21.45, 103.85, 104.05),
    "Hòa Bình": (20.75, 20.95, 105.25, 105.45),
    "Điện Biên": (21.35, 21.55, 103.00, 103.20),
    "Lai Châu": (22.35, 22.55, 103.40, 103.60),
    # Central Vietnam & Highlands
    "Thanh Hóa": (19.75, 20.00, 105.70, 105.90),
    "Nghệ An": (18.60, 18.85, 105.60, 105.80),
    "Hà Tĩnh": (18.30, 18.50, 105.85, 106.00),
    "Quảng Bình": (17.40, 17.60, 106.55, 106.70),
    "Quảng Trị": (16.70, 16.90, 107.10, 107.25),
    "Thừa Thiên Huế": (16.40, 16.55, 107.50, 107.65),
    "Đà Nẵng": (15.95, 16.15, 108.10, 108.30),
    "Quảng Nam": (15.50, 15.70, 108.40, 108.60),
    "Quảng Ngãi": (15.10, 15.25, 108.75, 108.90),
    "Bình Định": (13.70, 13.90, 109.15, 109.30),
    "Phú Yên": (13.05, 13.20, 109.25, 109.40),
    "Khánh Hòa": (12.20, 12.35, 109.15, 109.25),
    "Ninh Thuận": (11.50, 11.70, 108.95, 109.10),
    "Bình Thuận": (10.90, 11.05, 108.05, 108.20),
    "Kon Tum": (14.30, 14.45, 107.95, 108.05),
    "Gia Lai": (13.95, 14.10, 107.95, 108.10),
    "Đắk Lắk": (12.65, 12.80, 108.00, 108.15),
    "Đắk Nông": (11.95, 12.10, 107.65, 107.80),
    "Lâm Đồng": (11.90, 12.05, 108.40, 108.55),
    # Southern Vietnam
    "TP. Hồ Chí Minh": (10.65, 10.90, 106.55, 106.85),
    "Bình Dương": (10.90, 11.15, 106.60, 106.80),
    "Đồng Nai": (10.90, 11.10, 106.80, 107.05),
    "Bà Rịa - Vũng Tàu": (10.35, 10.55, 107.05, 107.25),
    "Tây Ninh": (11.25, 11.45, 106.05, 106.25),
    "Bình Phước": (11.50, 11.70, 106.85, 107.05),
    "Cần Thơ": (10.00, 10.15, 105.70, 105.85),
    "Long An": (10.50, 10.65, 106.35, 106.50),
    "Tiền Giang": (10.30, 10.45, 106.30, 106.45),
    "Bến Tre": (10.20, 10.30, 106.35, 106.45),
    "Trà Vinh": (9.90, 10.00, 106.30, 106.40),
    "Vĩnh Long": (10.20, 10.30, 105.90, 106.00),
    "Đồng Tháp": (10.40, 10.55, 105.60, 105.70),
    "An Giang": (10.35, 10.50, 105.40, 105.50),
    "Kiên Giang": (9.95, 10.15, 105.05, 105.20),
    "Hậu Giang": (9.75, 9.85, 105.45, 105.55),
    "Sóc Trăng": (9.55, 9.65, 105.95, 106.05),
    "Bạc Liêu": (9.25, 9.35, 105.70, 105.80),
    "Cà Mau": (9.15, 9.25, 105.10, 105.20),
}

PROVINCES_LIST = list(PROVINCES_GEO.keys())

DISTRICTS = [
    "Quận 1", "Quận 3", "Quận 5", "Quận 7", "Quận 10", "Quận Bình Thạnh", "Quận Tân Bình",
    "Quận Ba Đình", "Quận Hoàn Kiếm", "Quận Đống Đa", "Quận Cầu Giấy", "Quận Hai Bà Trưng", "Quận Hoàng Mai",
    "Quận Hải Châu", "Quận Thanh Khê", "Quận Sơn Trà", "Quận Ngũ Hành Sơn",
    "Quận Ninh Kiều", "Quận Bình Thủy", "Quận Cái Răng",
    "Quận Hồng Bàng", "Quận Ngô Quyền", "Quận Lê Chân",
    "Thành phố Thủ Đức", "Thành phố Dĩ An", "Thành phố Biên Hòa", "Thành phố Vũng Tàu", "Thành phố Hạ Long",
    "Thị xã Bến Cát", "Thị xã Sơn Tây", "Thị xã Duy Tiên", "Thị xã An Nhơn", "Thị xã Phú Mỹ",
    "Huyện Gia Lộc", "Huyện Châu Thành", "Huyện Thủy Nguyên", "Huyện Đông Anh", "Huyện Gia Lâm",
    "Huyện Phú Lộc", "Huyện Krông Pắc", "Huyện Đức Hòa", "Huyện Bình Chánh", "Huyện Củ Chi",
    "Phường Bến Nghé", "Phường Hàng Bông", "Phường Thạch Thang", "Phường Tân An", "Phường Dịch Vọng",
]

STREETS = [
    "Trần Hưng Đạo", "Lê Lợi", "Nguyễn Huệ", "Quang Trung", "Hùng Vương", "Hai Bà Trưng",
    "Phan Chu Trinh", "Điện Biên Phủ", "Lý Thường Kiệt", "Võ Nguyên Giáp", "Nguyễn Trãi",
    "Bạch Đằng", "Trường Chinh", "Cách Mạng Tháng Tám", "Hoàng Diệu", "Lê Duẩn",
    "Phạm Văn Đồng", "Võ Văn Kiệt", "Nguyễn Văn Cừ", "Kim Mã", "Cầu Giấy", "Giải Phóng",
    "Nguyễn Thị Minh Khai", "Nam Kỳ Khởi Nghĩa", "Pasteur", "Lê Thánh Tôn", "Tôn Đức Thắng",
    "Trần Phú", "Nguyễn Chí Thanh", "Hoàng Hoa Thám", "Xã Đàn", "Đại Cồ Việt", "Đinh Tiên Hoàng",
]

# Rich multi-scenario patterns for all 12 categories
CATEGORIES_PATTERNS = {
    "road_maintenance": {
        "unit": "infrastructure",
        "high": [
            ("Mặt đường sụt lún tạo hố tử thần sâu hơn 1m nguy cơ tai nạn chết người",
             "Tại trục đường chính xuất hiện hố sụt lún to như miệng giếng sâu hơn 1m không có biển cảnh báo hay rào chắn, nhiều người đi xe máy đã sập hố nguy hiểm tính mạng, đề nghị đơn vị quản lý đường bộ khẩn cấp phong tỏa và khắc phục ngay."),
            ("Mặt đường bị sạt lở đứt gãy sau mưa lũ nguy cơ sập cả đoạn đường",
             "Đoạn đường huyết mạch bị sạt lở nghiêm trọng phần ta luy âm, nứt toác toàn bộ mặt đường có thể đổ sập bất cứ lúc nào, nguy cơ chia cắt giao thông hoàn toàn, đề nghị kiểm tra xử lý khẩn cấp."),
            ("Hố ga mất nắp tạo hố sâu hoắm nguy hiểm tính mạng người tham gia giao thông",
             "Nắp cống hố ga giữa lòng đường bị xe tải đè vỡ sập xuống tạo hố sâu cực kỳ nguy hiểm vào ban đêm khi trời mưa ngập, đề nghị sửa chữa thay thế khẩn cấp tránh tai nạn chết người."),
            ("Đoạn đường lún sụt nứt toác hàm ếch nguy hiểm rình rập người đi xe",
             "Mặt đường rỗng chân tạo hàm ếch nguy hiểm kéo dài 5 mét, bề mặt chỉ còn lớp nhựa mỏng sắp sập, nhiều người đi đường suýt lọt hố, đề nghị cắm biển cảnh báo và gia cố khẩn cấp."),
            ("Vệt dầu nhớt đổ tràn mặt đường dài hàng trăm mét gây trượt ngã liên hoàn",
             "Xe bồn bị rò rỉ làm dầu nhớt chảy lênh láng trên mặt cầu và dốc cua, khiến hơn 10 xe máy bị trượt ngã liên tiếp, đề nghị lực lượng chức năng rải cát và xử lý ngay tránh tai nạn thảm khốc."),
            ("Mặt cầu bị gãy nhịp lún sụt khe co giãn trơ sắt nhọn nguy hiểm",
             "Khe co giãn trên mặt cầu bị gãy hở rộng gần 20cm, các thanh sắt nhọn nhô lên chém rách lốp xe và làm ngã người đi xe máy, đề nghị công ty quản lý cầu sửa chữa khẩn."),
        ],
        "normal": [
            ("Mặt đường xuất hiện nhiều ổ gà bong tróc đá dăm gây trượt ngã",
             "Đoạn đường trước khu dân cư bị xuống cấp nặng, xuất hiện chằng chịt ổ gà ổ voi, đá dăm văng tung tóe khiến người dân đi lại rất vất vả, đề nghị đơn vị duy tu dặm vá lại mặt đường."),
            ("Vỉa hè bị nứt vỡ lún sụt gạch lát đi lại khó khăn",
             "Tuyến vỉa hè dành cho người đi bộ bị gãy nứt nhiều viên gạch lát, tạo các rãnh mấp mô gây mất mỹ quan đô thị và dễ vấp ngã, kính đề nghị cơ quan chức năng cho sửa chữa."),
            ("Đường liên thôn nhiều đoạn mấp mô lồi lõm cần được bảo trì",
             "Tuyến đường liên thôn bị xe tải chạy nhiều tạo thành các vệt lún sâu, mùa mưa đọng nước lầy lội, kính mong chính quyền địa phương sớm có kế hoạch rải nhựa dặm vá."),
            ("Rãnh đào thi công cáp ngầm hoàn trả mặt đường cẩu thả lồi lõm",
             "Đơn vị thi công đào rãnh hạ ngầm cáp viễn thông nhưng khi hoàn trả chỉ đắp đất đá gồ ghề, không thảm lại nhựa bằng phẳng, đề nghị yêu cầu nhà thầu khắc phục đúng tiêu chuẩn."),
            ("Mặt đường bê tông nông thôn bị nứt nẻ bong tróc trơ sỏi đá",
             "Tuyến đường ngõ xóm sử dụng lâu năm đã bị mài mòn, xuất hiện các vệt nứt dọc kéo dài gây khó khăn cho việc đi lại của người cao tuổi, đề nghị hỗ trợ xi măng tu sửa."),
            ("Gờ giảm tốc trước cổng trường học bị bong tróc mài mòn mất tác dụng",
             "Các gờ cao su giảm tốc đã bị bung ốc vít, một số đoạn mất hẳn gờ làm xe cộ chạy qua phóng nhanh nguy hiểm, đề nghị bảo trì dặm vá lại."),
        ],
        "low": [
            ("Kiến nghị đưa đoạn đường vào kế hoạch nâng cấp thảm nhựa khi có kinh phí",
             "Kính gửi UBND xem xét đưa tuyến đường dân sinh vào danh mục đầu tư công trung hạn giai đoạn tới để thảm nhựa nâng cấp đồng bộ, tạo điều kiện phát triển kinh tế địa phương."),
            ("Đề xuất chỉnh trang lại vỉa hè đồng bộ với khu phố kiểu mẫu",
             "Người dân kiến nghị cơ quan chức năng khảo sát thay thế hệ thống gạch lát vỉa hè cũ bằng đá tự nhiên đồng bộ với cảnh quan tuyến phố mới."),
            ("Góp ý khảo sát làm gờ giảm tốc tại khu vực gần cổng trường học",
             "Đề xuất ngành giao thông xem xét bố trí thêm gờ giảm tốc và vạch sơn cảnh báo khu vực trường học nhằm tăng cường an toàn khi các cháu học sinh tan trường."),
            ("Kiến nghị mở rộng đoạn cua hẹp ngõ xóm để xe ô tô cứu hỏa tiếp cận",
             "Kính gửi phòng đô thị quận khảo sát nắn thẳng khúc cua góc ngõ nhằm bảo đảm bán kính quay xe cho phương tiện cứu nạn cứu hộ khi có sự cố."),
            ("Đề xuất lắp đặt thêm biển chỉ dẫn tên ngõ ngách rõ ràng hơn",
             "Người dân kiến nghị lắp đặt biển báo số nhà và tên ngõ ngách bằng phản quang để người giao hàng và người lạ dễ tìm đường."),
        ],
    },
    "drainage_sewage": {
        "unit": "infrastructure",
        "high": [
            ("Nắp cống bị cuốn trôi nước xoáy cực mạnh nguy cơ hút người",
             "Cống thoát nước chính bị bật nắp trôi mất, dòng nước chảy xiết cuồn cuộn tạo vùng xoáy nguy hiểm chết người, đề nghị cơ quan chức năng khẩn cấp rào chắn và xử lý ngay."),
            ("Ngập sâu nghiêm trọng hơn nửa mét cô lập toàn bộ khu dân cư",
             "Hệ thống cống chính bị tắc khiến nước dâng cao ngập sâu hơn 60cm tràn vào nhà dân, đe dọa chập điện và gây hư hỏng tài sản nghiêm trọng, đề nghị hỗ trợ cứu hộ bơm thoát nước khẩn cấp."),
            ("Cống xả hóa chất độc hại bốc mùi nồng nặc gây ngất xỉu nguy cấp",
             "Đường cống bất ngờ tràn ra chất lỏng đen ngòm sủi bọt bốc mùi axit hắc nồng nặc khiến nhiều người dân trong ngõ khó thở, buồn nôn, đề nghị lực lượng chức năng khẩn cấp kiểm tra."),
            ("Vỡ đường ống thoát nước thải áp lực làm sụt lún chân móng nhà dân",
             "Đường ống dẫn nước thải ngầm bị bục vỡ áp lực lớn, nước đen xối xả làm xói mòn sụt lún chân móng hai ngôi nhà liền kề, nguy cơ nứt đổ nhà khẩn cấp, đề nghị khắc phục ngay."),
            ("Nước cống trào ngược ngập lênh láng khu cấp cứu bệnh viện",
             "Mưa to làm hệ thống thoát nước bên ngoài tắc nghẽn, nước cống đen ngòm tràn vào tầng 1 khu khám chữa bệnh, đề nghị điều xe hút bể phốt khẩn cấp hỗ trợ."),
        ],
        "normal": [
            ("Cống thoát nước tắc nghẽn gây ngập úng cục bộ sau mưa",
             "Mỗi khi có mưa rào là tuyến đường ngập sâu khoảng 20-30cm do rác bịt kín miệng hố ga thoát nước, đề nghị công ty thoát nước cho công nhân xuống nạo vét khơi thông."),
            ("Miệng cống bốc mùi hôi thối nồng nặc ảnh hưởng sinh hoạt",
             "Miệng hố ga trước cửa nhà dân bị hỏng van ngăn mùi, mùi hôi thối xông lên cả ngày đêm không chịu nổi, đề nghị lắp đặt van ngăn mùi mới."),
            ("Nắp hố ga bị nứt vỡ kênh lệch phát tiếng kêu mỗi khi xe chạy qua",
             "Nắp cống bằng gang bị gãy một góc và kênh vênh, mỗi lần phương tiện chạy qua phát ra tiếng va đập rất lớn gây mất giấc ngủ, đề nghị căn chỉnh thay thế."),
            ("Nước thải sinh hoạt chảy tràn ra mặt đường do tắc đường ống nhánh",
             "Ống cống nhánh của ngõ dân cư bị đầy bùn đất, nước giặt xả tràn lênh láng ra ngõ gây trơn trượt và ô nhiễm, đề nghị hỗ trợ thông tắc cống."),
            ("Miệng hố thu nước mưa bị đất đá lấp đầy không thoát được nước",
             "Hố thu nước ven đường bị cát bụi công trình bồi lắng dày đặc làm giảm 80% khả năng thu nước, đề nghị nạo vét trước đợt mưa lớn tới."),
        ],
        "low": [
            ("Đề xuất khảo sát nạo vét hệ thống cống rãnh định kỳ trước mùa mưa bão",
             "Kiến nghị công ty môi trường đô thị lên kế hoạch nạo vét bùn đất toàn bộ hệ thống thoát nước trên địa bàn phường trước tháng 8 để chủ động chống ngập úng."),
            ("Kiến nghị đầu tư xây dựng hệ thống cống ngầm thay thế rãnh thoát nước hở",
             "Đề xuất chính quyền địa phương đưa vào dự án cải tạo đô thị, thay thế các rãnh thoát nước hở bốc mùi bằng cống hộp ngầm có nắp đậy mỹ quan."),
            ("Góp ý giải pháp tuyên truyền người dân không vứt rác bịt miệng cống",
             "Đề nghị tổ dân phố phối hợp ban ngành tuyên truyền, nhắc nhở các hộ kinh doanh không đổ dầu mỡ và quét rác xuống miệng cống thoát nước."),
            ("Đề xuất lắp đặt lưới chắn rác kiểu mới tại các hố ga mặt đường",
             "Kiến nghị sử dụng thiết bị thu rác có giỏ gom thông minh để công nhân môi trường dễ dàng nâng lên làm sạch mà không mất nhiều công sức."),
        ],
    },
    "street_lighting": {
        "unit": "infrastructure",
        "high": [
            ("Chập điện phát tia lửa sáng rực trên cột đèn chiếu sáng nguy cơ cháy nổ",
             "Hộp đấu nối điện trên cột đèn chiếu sáng phát nổ lách tách và bốc khói đen kèm tia lửa điện rơi xuống đường, đề nghị ngắt điện và sửa chữa khẩn cấp tránh cháy lan."),
            ("Dây điện chiếu sáng đứt rơi lòng thòng xuống vỉa hè hở điện nguy hiểm",
             "Dây cáp điện chiếu sáng đô thị bị đứt rơi sà xuống sát mặt đất, trời lại đang mưa phùn nguy cơ giật điện chết người rất cao, đề nghị xử lý khẩn cấp."),
            ("Cột đèn chiếu sáng bị nghiêng nứt toác chân cột sắp đổ đè đường",
             "Cột đèn chiếu sáng bằng thép bị xe tải va chạm nghiêng 45 độ, chân cột nứt gãy đe dọa đổ ập xuống đường bất cứ lúc nào, đề nghị khẩn cấp khắc phục."),
            ("Rò rỉ điện ra vỏ cột đèn chiếu sáng kim loại khi trời mưa ngập",
             "Người dân chạm vào cột đèn thì bị tê giật mạnh, kiểm tra bằng bút thử điện thấy sáng đỏ, cực kỳ nguy hiểm cho học sinh tan học, đề nghị ngắt điện khẩn cấp."),
        ],
        "normal": [
            ("Đèn đường ngõ xóm bị cháy bóng không sáng suốt tuần qua",
             "Hệ thống đèn đường ngõ 45 bị hỏng bóng tối om cả tuần nay, gây mất an ninh trật tự và tiềm ẩn nguy cơ tai nạn cho người già trẻ nhỏ, đề nghị thay bóng đèn mới."),
            ("Đèn chiếu sáng chập chờn lúc sáng lúc tắt liên tục",
             "Bóng đèn cao áp trên tuyến phố bị chập chờn nhấp nháy liên tục không đảm bảo ánh sáng vào ban đêm, đề nghị đơn vị chiếu sáng kiểm tra tăng phô khởi động."),
            ("Ánh sáng đèn đường quá mờ không đủ tầm nhìn lưu thông",
             "Bóng đèn vàng cũ công suất yếu lại bị cây che khuất khiến đoạn đường rất tối, đề nghị thay bóng đèn công suất phù hợp hơn."),
            ("Đèn đường bật sáng cả ban ngày gây lãng phí điện năng",
             "Hệ thống rơ-le hẹn giờ chiếu sáng bị lỗi khiến cả dãy đèn cao áp bật suốt từ trưa đến chiều, đề nghị kỹ thuật căn chỉnh lại rơ le tự động."),
            ("Cửa hộp kỹ thuật dưới chân cột đèn bị bung nắp hở dây điện",
             "Nắp đậy hộp kỹ thuật cột điện bị mất ốc rơi ra, để lộ các mối nối dây điện bên trong ngõ xóm có nhiều trẻ con chơi đùa, đề nghị đậy nắp an toàn."),
        ],
        "low": [
            ("Kiến nghị lắp bổ sung thêm bóng đèn chiếu sáng ngõ mới mở",
             "Kính mong công ty chiếu sáng đô thị xem xét bổ sung thêm 2 điểm đèn chiếu sáng tại đoạn ngõ mới mở rộng để bà con đi lại thuận tiện hơn."),
            ("Đề xuất thay thế hệ thống đèn chiếu sáng cũ sang đèn LED tiết kiệm điện",
             "Đề xuất thành phố có lộ trình thay thế đèn thủy ngân cao áp cũ bằng đèn LED thông minh nhằm tiết kiệm điện năng và thân thiện môi trường."),
            ("Góp ý điều chỉnh giờ bật tắt đèn chiếu sáng công cộng theo mùa",
             "Kiến nghị trung tâm điều hành chiếu sáng đô thị điều chỉnh giờ bật đèn mùa đông sớm hơn 30 phút vì trời tối rất nhanh."),
            ("Đề xuất lắp đặt đèn năng lượng mặt trời tại các khu đất trống công cộng",
             "Kiến nghị thí điểm hệ thống đèn chiếu sáng solar tại các hoa viên nhỏ và lối đi dạo bờ sông."),
        ],
    },
    "waste_management": {
        "unit": "environment",
        "high": [
            ("Bãi rác tự phát bốc cháy dữ dội khói độc mịt mù đe dọa cháy lan khu dân cư",
             "Bãi rác trái phép đang bốc cháy dữ dội, tàn lửa bay tứ tung gần kho hàng và nhà dân xung quanh, khói độc ngột ngạt bao trùm, đề nghị lực lượng PCCC và chính quyền dập lửa khẩn cấp."),
            ("Đổ trộm rác thải y tế kim tiêm và chất thải nguy hại tràn ra đường",
             "Đối tượng xấu đổ trộm hàng bao tải rác thải y tế kim tiêm dính máu và lọ thuốc nguy hại ngay lối đi, nguy cơ lây nhiễm bệnh tật khẩn cấp, đề nghị kiểm tra thu gom và truy tìm thủ phạm."),
            ("Rác thải chặn kín dòng chảy kênh mương gây ngập úng và ô nhiễm nghiêm trọng",
             "Rác thải sinh hoạt ùn ứ kết thành mảng lớn bịt kín cống thoát lũ của xã, nước dâng tràn bờ đe dọa ngập hoa màu và ô nhiễm nguồn nước sinh hoạt."),
            ("Đổ trộm thùng hóa chất bốc khói cay mắt ra bãi rác dân sinh",
             "Xe tải lạ đổ trộm 3 thùng phuy nhựa rò rỉ chất lỏng màu vàng bốc khói cay xè mắt mũi khiến cả khu xóm ho sặc sụa, đề nghị cơ quan công an môi trường xử lý khẩn cấp."),
        ],
        "normal": [
            ("Rác thải sinh hoạt ùn ứ 3 ngày không ai thu gom bốc mùi khó chịu",
             "Xe rác không vào lấy rác suốt từ đầu tuần khiến rác chất đống trước ngõ tràn ra lòng đường bốc mùi hôi thối nồng nặc, đề nghị công ty môi trường nhanh chóng cho xe gom."),
            ("Thùng rác công cộng bị hỏng vỡ nắp rác rơi vãi xung quanh",
             "Thùng chứa rác công cộng bị bể đáy và mất nắp đậy, nước rác rỉ ra đường thu hút ruồi nhặng gây mất vệ sinh, đề nghị thay thùng rác mới."),
            ("Đổ trộm xà bần phế thải xây dựng chiếm dụng lòng lề đường",
             "Xe ba gác đổ trộm đất đá bê tông thừa trên vỉa hè gây cản trở người đi bộ và mất mỹ quan đô thị, đề nghị chính quyền xử lý bốc dọn."),
            ("Điểm tập kết rác để nước rỉ rác đen ngòm chảy lênh láng mặt đường",
             "Vị trí ép rác tạm thời không có hố thu gom nước thải, nước rỉ rác đen ngòm hôi tanh bám chặt mặt đường gây trơn ngã, đề nghị vệ sinh xịt rửa sạch sẽ."),
            ("Người dân vứt xác súc vật thối rữa ven đường bốc mùi hôi thối",
             "Xác động vật chết bị vứt trong bao tải ven đường bốc mùi xú uế nồng nặc, đề nghị đội vệ sinh môi trường chôn cất xử lý khử trùng."),
        ],
        "low": [
            ("Đề xuất đặt thêm thùng rác phân loại tại công viên công cộng",
             "Kiến nghị ban quản lý công viên trang bị thêm các thùng rác phân loại 3 ngăn (hữu cơ, tái chế, vô cơ) để nâng cao ý thức giữ gìn vệ sinh chung."),
            ("Kiến nghị điều chỉnh giờ thu gom rác tránh khung giờ cao điểm tan tầm",
             "Đề xuất công ty môi trường chuyển lịch thu gom rác từ 17h sang sau 19h để tránh gây ùn tắc giao thông vào giờ người dân đi làm về."),
            ("Góp ý phát động phong trào ngày chủ nhật xanh dọn vệ sinh khu phố",
             "Đề xuất đoàn thanh niên và hội phụ nữ phường tổ chức ra quân dọn dẹp các điểm đen rác thải định kỳ hàng tháng."),
            ("Kiến nghị lắp camera phạt nguội hành vi xả rác bừa bãi nơi công cộng",
             "Đề xuất lắp đặt camera giám sát tại các bãi đất trống hay bị đổ rác trộm để xử phạt nghiêm theo quy định."),
        ],
    },
    "pollution_emissions": {
        "unit": "environment",
        "high": [
            ("Rò rỉ khí độc hại từ nhà máy bốc mùi nồng nặc gây khó thở ngất xỉu",
             "Khu công nghiệp phát tán mùi hóa chất clo hắc nồng nặc khiến nhiều học sinh trường gần đó bị nôn mửa, khó thở phải đưa đi cấp cứu, đề nghị cơ quan môi trường đình chỉ và xử lý khẩn cấp."),
            ("Lò đốt xả khói đen kịt ngột ngạt bao trùm cả khu dân cư nguy cơ ngộ độc",
             "Cơ sở tái chế nhựa hoạt động lén lút lúc nửa đêm xả cột khói đen đặc khét lẹt nồng nặc mùi dioxin bao trùm phòng ngủ người dân, đề nghị công an môi trường ập vào bắt quả tang ngay lập tức."),
            ("Xả trộm nước thải hóa chất đen ngòm sủi bọt ra nguồn nước sinh hoạt",
             "Nước thải công nghiệp chưa qua xử lý sủi bọt trắng xóa bốc mùi hôi thối xả thẳng ra kênh tưới tiêu nông nghiệp làm cá chết hàng loạt, đề nghị niêm phong cống xả."),
            ("Cháy kho chứa thuốc bảo vệ thực vật phát tán mùi độc cực mạnh",
             "Đám cháy tại kho vật tư nông nghiệp làm khói chứa hơi thuốc trừ sâu nồng nặc lan rộng khắp xã, đe dọa nhiễm độc hô hấp toàn bộ cư dân, đề nghị sơ tán và ứng phó khẩn cấp."),
        ],
        "normal": [
            ("Quán ăn xả khói nướng thịt mù mịt cả ngày vào nhà dân xung quanh",
             "Quán nướng không lắp đặt ống hút khói đạt chuẩn, khói mỡ và bụi than thổi trực tiếp vào phòng khách các nhà lân cận, đề nghị phường kiểm tra chấn chỉnh."),
            ("Công trình xây dựng che chắn sơ sài làm bụi bay mù mịt cả con đường",
             "Công trình thi công nhà ở không dùng lưới che bụi, xe chở vật liệu làm rơi vãi cát sỏi bụi mù mịt mỗi khi có xe chạy qua, đề nghị nhắc nhở che chắn."),
            ("Xưởng gia công cơ khí xả mùi sơn PU nồng nặc ra khu phố",
             "Xưởng mộc phun sơn đồ gỗ trong ngõ hẹp không có buồng lọc mùi, mùi sơn phát tán nồng nặc làm trẻ nhỏ ho liên tục, đề nghị di dời buồng sơn."),
            ("Trang trại lợn xả nước phân chưa qua hầm biogas hôi thối cả làng",
             "Hộ chăn nuôi xả tràn nước thải phân lợn xuống rãnh thoát nước chung bốc mùi hôi nồng nặc suốt mùa hè, đề nghị xử phạt yêu cầu làm bể biogas."),
            ("Cơ sở giặt là xả bọt xà phòng hóa chất tràn ra ngõ đi chung",
             "Nước giặt công nghiệp sủi bọt trắng xóa có mùi clo nồng nặc chảy lênh láng mặt ngõ làm ảnh hưởng môi trường sống."),
        ],
        "low": [
            ("Kiến nghị lắp đặt trạm quan trắc tự động đo chất lượng không khí",
             "Đề xuất thành phố bổ sung thêm trạm cảm biến đo bụi mịn PM2.5 tại nút giao thông đông đúc để người dân theo dõi chỉ số AQI hàng ngày."),
            ("Đề xuất kế hoạch di dời các cơ sở sản xuất gây ô nhiễm ra khỏi khu dân cư",
             "Kính gửi Sở Tài nguyên Môi trường lộ trình di dời các cơ sở tiểu thủ công nghiệp đan xen trong khu dân cư vào cụm công nghiệp tập trung theo quy hoạch."),
            ("Góp ý tăng cường thanh tra định kỳ việc chấp hành bảo vệ môi trường",
             "Đề xuất chính quyền địa phương lập đoàn kiểm tra liên ngành kiểm tra việc đăng ký giấy phép môi trường của các cơ sở kinh doanh dịch vụ."),
            ("Đề xuất khuyến khích các hộ kinh doanh ẩm thực chuyển đổi sang bếp điện không khói",
             "Kiến nghị phường có chính sách hỗ trợ và tuyên truyền các quán nướng ven đường thay thế bếp than tổ ong."),
        ],
    },
    "noise_disturbance": {
        "unit": "environment",
        "high": [
            ("Tụ tập mở loa công suất lớn thách thức bạo lực đe dọa an ninh trật tự",
             "Nhóm đối tượng tụ tập ăn nhậu bật loa kẹo kéo hết cỡ từ chiều đến nửa đêm, khi hàng xóm sang nhắc nhở thì chửi bới mang dao gậy đe dọa chém, đề nghị công an can thiệp khẩn cấp."),
            ("Quán bar hoạt động chui phát tiếng bass rung chuyển nhà cửa phá rối trật tự",
             "Cơ sở kinh doanh trá hình mở nhạc vũ trường chát chúa rung lắc cả tường nhà dân xung quanh suốt từ 23h đến 3h sáng, thách thức chính quyền, đề nghị đình chỉ ngay."),
            ("Công trường thi công ép cọc rung chấn nứt tường nhà dân giữa đêm khuya",
             "Công trình thi công trái phép ban đêm dùng búa rung công suất lớn làm rung chuyển nền móng nứt toác tường nhà hàng xóm nguy cơ đổ sập, đề nghị lập biên bản tạm dừng."),
            ("Xưởng cưa xẻ đá granite hoạt động 24/24 gây tiếng rít đinh tai nhức óc",
             "Máy cắt đá công suất cực lớn gầm rú suốt đêm khuya không cho ai chợp mắt, người dân bức xúc sắp xảy ra xô xát đổ máu, đề nghị công an giải quyết khẩn cấp."),
        ],
        "normal": [
            ("Hát karaoke loa kéo quá giờ quy định gây mất ngủ cho người dân",
             "Hộ gia đình mở loa kéo hát karaoke âm lượng lớn liên tục đến tận 23h đêm, làm người già không ngủ được và trẻ em không học bài được, đề nghị tổ trưởng nhắc nhở."),
            ("Quán nhậu mở nhạc sôi động quá to ảnh hưởng khu dân cư",
             "Quán bia đối diện bật nhạc ầm ĩ để thu hút khách làm phiền toái cả khu phố, đề nghị phường yêu cầu vặn nhỏ âm lượng đúng quy định."),
            ("Công trường thi công gây tiếng ồn lớn vào khung giờ nghỉ trưa",
             "Máy trộn bê tông và máy cưa sắt hoạt động ầm ĩ từ 12h đến 13h30 trưa làm xáo trộn giờ nghỉ ngơi của người dân, đề nghị tuân thủ khung giờ làm việc."),
            ("Tiệm sửa xe máy nẹt pô thử máy gây ồn ào cả ngày trong ngõ",
             "Tiệm độ xe liên tục thử pô xe máy nổ đì đùng chát chúa trong khu dân cư yên tĩnh, đề nghị công an phường kiểm tra giấy phép kinh doanh."),
            ("Xe chở loa quảng cáo bán hàng rong đi lại rao ầm ĩ liên tục",
             "Xe bán hàng rong gắn loa phóng thanh phát đi phát lại điệp khúc bán hàng với âm lượng inh ỏi làm phiền khu dân cư."),
        ],
        "low": [
            ("Kiến nghị ban hành quy chế khu dân cư văn minh không tiếng ồn",
             "Đề xuất UBND phường phát động phong trào xây dựng ngõ xóm yên tĩnh, quy định rõ khung giờ hạn chế tiếng ồn và đưa vào tiêu chí bình xét gia đình văn hóa."),
            ("Đề xuất tuyên truyền các cơ sở kinh doanh cam kết kiểm soát âm lượng",
             "Kiến nghị tổ trật tự đô thị cho các quán cà phê, nhà hàng ký cam kết lắp đặt vật liệu cách âm nếu có biểu diễn âm nhạc."),
            ("Góp ý mức xử phạt vi phạm hành chính đối với hành vi gây ô nhiễm tiếng ồn",
             "Đề nghị các lực lượng chức năng thường xuyên kiểm tra bằng thiết bị đo tiếng ồn chuyên dụng để có căn cứ xử phạt khách quan."),
            ("Đề xuất quy hoạch khu vui chơi âm nhạc tách biệt khu dân cư sinh sống",
             "Kiến nghị địa phương định hướng các tụ điểm âm nhạc về khu phố đi bộ tập trung."),
        ],
    },
    "urban_trees": {
        "unit": "environment",
        "high": [
            ("Cây cổ thụ bật gốc gãy đổ đè trúng người và đường dây điện cực kỳ nguy kịch",
             "Cây xà cừ lớn bị giông lốc bật gốc đổ ập đè bẹp xe máy người đi đường và kéo đứt toàn bộ đường điện hạ thế, đề nghị cứu hộ và khắc phục khẩn cấp giải cứu nạn nhân."),
            ("Cành cây đại thụ mục rỗng nghiêng sắp sập xuống mái tôn nhà dân",
             "Nhánh cây bằng lăng cổ thụ đường kính 40cm bị mục gốc rỗng ruột kêu răng rắc khi có gió, đe dọa rơi đè sập căn nhà có người già ở bên dưới, đề nghị cưa hạ khẩn cấp."),
            ("Cây đổ chắn ngang cầu huyết mạch gây tê liệt giao thông hoàn toàn",
             "Cây phượng vĩ to gãy đổ chắn trọn lối lên cầu, các phương tiện không thể di chuyển gây kẹt xe hàng km, đề nghị công ty cây xanh cử xe cẩu đến cưa dọn ngay."),
            ("Cây đổ đè sập cột điện chiếu sáng làm lửa bốc cháy ngùn ngụt",
             "Cây bàng lớn bật gốc đè gãy đôi cột điện gây nổ biến áp và cháy bụi rậm xung quanh, đe dọa cháy lan nhà dân lân cận, đề nghị PCCC và cây xanh ứng cứu khẩn cấp."),
        ],
        "normal": [
            ("Cành cây phát triển um tùm vướng vào dây viễn thông và chiếu sáng",
             "Tán cây xanh trước nhà phát triển quá rậm rạp đè nặng lên các búi cáp viễn thông, đề nghị công nhân cây xanh đến cắt tỉa bảo đảm an toàn."),
            ("Cây xanh che khuất tầm nhìn biển báo và đèn tín hiệu giao thông",
             "Nhánh cây xòe rộng che kín biển báo cấm và đèn giao thông tại ngã ba khiến tài xế không quan sát được, đề nghị tỉa bớt cành che khuất."),
            ("Rễ cây cổ thụ trồi lên làm nứt vỡ mặt vỉa hè người đi bộ",
             "Rễ cây to phát triển phá vỡ mặt đường gạch vỉa hè, tạo gờ mấp mô khiến người đi bộ hay bị vấp té, đề nghị kiểm tra hạ rễ và chỉnh trang lại."),
            ("Cây xanh bị sâu róm bọ nẹp cắn trụi lá rơi rụng gây ngứa ngáy",
             "Dãy cây hoa sữa trên phố bị sâu bọ tấn công rơi đầy xuống người đi bộ gây mẩn ngứa khó chịu, đề nghị phun thuốc trừ sâu sinh học."),
            ("Cây chết khô lâu ngày đứng trơ trọi giữa phố cần chặt hạ",
             "Cây bóng mát bị chết khô cành nhánh giòn gãy rơi rụng mỗi khi có gió to, đề nghị chặt bỏ và trồng cây mới thay thế."),
        ],
        "low": [
            ("Đề xuất trồng bổ sung cây hoa ban tạo cảnh quan tuyến phố xanh đẹp",
             "Kính gửi công ty công viên cây xanh xem xét trồng thêm cây có hoa như giáng hương, kèn hồng trên dải phân cách để tạo điểm nhấn du lịch cho thành phố."),
            ("Kiến nghị lập kế hoạch cắt tỉa cành cây định kỳ hàng năm trước mùa mưa bão",
             "Đề xuất UBND phường rà soát toàn bộ cây xanh trên các trục ngõ, lên lịch chặt tỉa đồng loạt trong tháng 7 để chủ động phòng chống bão."),
            ("Góp ý thay thế giống cây ăn quả rụng hạt trơn trượt bằng cây đô thị chuyên dụng",
             "Kiến nghị thay thế các cây si, cây sung ven đường bằng giống cây xanh đô thị rễ cọc ít gãy cành và không làm bẩn mặt đường."),
            ("Đề xuất xã hội hóa trồng hoa và cây cảnh quanh các gốc cây xanh vỉa hè",
             "Người dân kiến nghị cho phép các tổ dân phố trồng viền hoa quanh bồn cây để làm đẹp mỹ quan đô thị."),
        ],
    },
    "traffic_signals": {
        "unit": "transport",
        "high": [
            ("Đèn giao thông ngã tư tê liệt hoàn toàn gây va chạm liên hoàn hỗn loạn",
             "Cụm đèn tín hiệu giao thông nút giao trọng điểm bị tắt ngóm vào giờ cao điểm, các luồng xe lao vào nhau gây tai nạn liên tiếp và tê liệt nút giao, đề nghị CSGT và kỹ thuật đến xử lý khẩn cấp."),
            ("Đèn tín hiệu bị kẹt cả 4 hướng đều bật đèn xanh cực kỳ nguy hiểm",
             "Tủ điều khiển bị chập mạch khiến cả hai chiều giao cắt cùng sáng đèn xanh, các phương tiện phóng nhanh lao thẳng vào nhau suýt đâm trực diện, đề nghị ngắt điện chuyển đèn vàng cảnh báo ngay."),
            ("Biển báo cấm ngược chiều bị gãy rơi trơ cọc nhọn đâm người đi đường",
             "Biển báo phân làn bị xe đâm gãy gập, thanh sắt cọc biển chĩa xiên ra làn xe máy gây nguy hiểm va quẹt cho người đi đường, đề nghị tháo dỡ xử lý khẩn cấp."),
            ("Cột đèn tín hiệu giao thông bị xe container đâm nghiêng sắp đổ đè xe máy",
             "Cột đèn tín hiệu bằng sắt to lớn bị xe tông bung chân đế nghiêng 60 độ chực chờ đổ sập đè lên người dừng đèn đỏ bên dưới, đề nghị cứu hộ và giữ cố định ngay lập tức."),
        ],
        "normal": [
            ("Đèn tín hiệu giao thông nhấp nháy đèn vàng cả ngày lẫn đêm",
             "Đèn giao thông tại ngã tư bị chuyển sang chế độ nhấp nháy vàng suốt từ hôm qua đến nay khiến xe cộ qua lại lộn xộn không ai nhường ai, đề nghị chuyển lại chu kỳ hoạt động bình thường."),
            ("Đèn đỏ bị cháy bóng chỉ sáng đèn xanh và đèn vàng gây hoang mang",
             "Bóng đèn LED màu đỏ của trụ đèn bị hỏng không sáng, người tham gia giao thông không biết khi nào phải dừng lại rất dễ bị phạt oan và va chạm, đề nghị thay cụm LED đỏ."),
            ("Thời lượng đèn xanh quá ngắn chỉ 10 giây gây ùn tắc kéo dài",
             "Lượng phương tiện lưu thông trên trục chính rất đông nhưng đèn xanh chỉ mở 12 giây trong khi đèn đỏ tới 60 giây, đề nghị điều chỉnh lại chu kỳ pha đèn hợp lý hơn."),
            ("Biển báo phân làn giao thông bị cây xanh che khuất hoàn toàn",
             "Biển báo cấm rẽ trái bị tán lá cây rậm rạp che kín khiến các tài xế tỉnh ngoài không nhìn thấy nên hay vi phạm, đề nghị tỉa cành hoặc dời biển ra vị trí thoáng."),
            ("Vạch sơn kẻ đường cho người đi bộ bị mờ hoàn toàn",
             "Vạch sang đường trước cổng bệnh viện bị mòn trơ đáy nhựa đường, tài xế không giảm tốc nhường đường cho bệnh nhân đi bộ, đề nghị sơn lại vạch phản quang nhiệt dẻo."),
        ],
        "low": [
            ("Kiến nghị lắp thêm đèn tín hiệu cho người đi bộ bấm nút sang đường",
             "Đề xuất ngành giao thông bố trí cụm đèn tín hiệu nút bấm dành cho người đi bộ tại khu vực trước trường đại học để học sinh sinh viên qua đường an toàn."),
            ("Đề xuất lắp đặt đồng hồ đếm ngược giây tại nút giao thông",
             "Kiến nghị bổ sung bảng hiển thị số đếm lùi thời gian tại các trụ đèn để lái xe chủ động dừng xe an toàn, hạn chế phanh gấp."),
            ("Góp ý điều chỉnh chu kỳ đèn tín hiệu linh hoạt theo mật độ giao thông thông minh",
             "Đề xuất thành phố ứng dụng camera AI đo đếm lưu lượng phương tiện để tự động điều chỉnh chu kỳ đèn giao thông theo thời gian thực."),
            ("Kiến nghị sơn bổ sung vạch kẻ mắt võng chống ùn tắc tại lối rẽ phải",
             "Đề xuất sơn ô mắt võng màu vàng tại vị trí cho phép rẽ phải liên tục để các xe đi thẳng không dừng đỗ chắn lối."),
        ],
    },
    "traffic_congestion_safety": {
        "unit": "transport",
        "high": [
            ("Nhóm đua xe lạng lách nẹt pô gây tai nạn chết người bỏ chạy",
             "Hàng chục thanh niên tụ tập đua xe trái phép nẹt pô inh ỏi lạng lách tông thẳng vào người đi đường gây chấn thương sọ não rồi rồ ga bỏ chạy, đề nghị lực lượng cảnh sát vây bắt khẩn cấp."),
            ("Xe tải chở cát sỏi làm rơi vãi đá tảng xuống đường gây tai nạn liên tiếp",
             "Xe ben cơi nới thành thùng làm rơi các khối đá hộc sắc nhọn to bằng nắm tay rải khắp làn xe cơ giới, nhiều xe máy đâm trúng ngã nhào ra đường, đề nghị truy tìm và dọn dẹp khẩn cấp."),
            ("Tê liệt toàn bộ giao thông cửa ngõ do xe đầu kéo lật chắn ngang đường",
             "Xe container bị lật ngang chắn trọn cả 3 làn xe huyết mạch, hàng nghìn phương tiện chôn chân kẹt cứng nhiều giờ trong thời tiết nắng gắt, đề nghị điều xe cẩu cứu hộ khẩn cấp."),
            ("Xe bồn chở xăng dầu bốc khói nghi ngút tại nút giao đông đúc",
             "Xe bồn có dấu hiệu kẹt phanh phát lửa và bốc khói mịt mù ở bánh sau giữa dòng phương tiện ùn ứ, nguy cơ cháy nổ kinh hoàng, đề nghị CSGT và PCCC can thiệp khẩn cấp."),
        ],
        "normal": [
            ("Ùn tắc giao thông cục bộ giờ cao điểm nút giao do không có người điều tiết",
             "Vào mỗi buổi sáng và chiều tan tầm, nút giao này thường xuyên bị kẹt cứng xung đột các hướng đi, đề nghị bố trí lực lượng CSGT hoặc dân quân điều tiết hướng dẫn."),
            ("Xe khách dừng đón trả khách tùy tiện ngay dốc cầu gây mất an toàn",
             "Các xe khách liên tỉnh liên tục dừng đỗ trái phép giữa đường để bắt khách và bốc dỡ hàng hóa làm cản trở dòng xe phía sau, đề nghị phạt nguội qua camera."),
            ("Xe quá tải né trạm cân đi vào đường dân sinh phá nát đường xóm",
             "Đoàn xe tải nặng chở đất đá né trạm thu phí đi vào đường liên thôn nhỏ hẹp làm bụi bay mù mịt và đe dọa an toàn cho trẻ em đi học, đề nghị cắm biển hạn chế tải trọng."),
            ("Hàng quán họp chợ tự phát lấn trọn lòng đường giờ tan tầm",
             "Tiểu thương bày bán rau củ thịt cá tràn ra 2/3 mặt đường vào buổi chiều khiến các phương tiện qua lại rất khó khăn, đề nghị lực lượng trật tự giải tỏa."),
            ("Điểm quay đầu xe bất hợp lý gây xung đột dòng phương tiện",
             "Điểm mở dải phân cách quá gần ngã tư khiến các xe quay đầu cắt mặt dòng xe đi thẳng gây ùn ứ nghiêm trọng, đề nghị dịch chuyển điểm quay đầu xa hơn."),
        ],
        "low": [
            ("Kiến nghị cắm biển cấm đỗ ngày chẵn lẻ để giảm ùn ứ tuyến phố hẹp",
             "Đề xuất Sở Giao thông xem xét cắm biển cấm đỗ xe theo ngày chẵn và ngày lẻ trên trục phố nhỏ nhằm tạo sự thông thoáng cho các phương tiện lưu thông."),
            ("Đề xuất mở rộng nút giao thông thắt cổ chai để tăng năng lực lưu thoát xe",
             "Kính gửi thành phố kế hoạch giải phóng mặt bằng mở rộng đoạn cua hẹp để giảm tải áp lực giao thông trong tương lai."),
            ("Góp ý điều chỉnh tổ chức giao thông một chiều đối với tuyến phố nhỏ",
             "Đề xuất phân luồng một chiều đối với xe ô tô vào các khung giờ cao điểm để tránh tình trạng hai xe tránh nhau gây tắc đường kéo dài."),
            ("Kiến nghị bố trí thêm làn dành riêng cho xe đạp trên các tuyến đại lộ",
             "Đề xuất xây dựng hệ thống đường riêng cho xe đạp đô thị nhằm khuyến khích lối sống xanh và giảm phát thải."),
        ],
    },
    "sidewalk_encroachment": {
        "unit": "transport",
        "high": [
            ("Quán nhậu bày bàn ghế chiếm trọn vỉa hè đánh người nhắc nhở thương tích",
             "Chủ quán nhậu kê kín bàn ghế chiếm toàn bộ vỉa hè và lòng đường, khi người dân nhắc nhở thì nhân viên quán xông vào hành hung đấm đá gây thương tích chảy máu, đề nghị công an xử lý hình sự khẩn cấp."),
            ("Cơi nới rào chắn bằng sắt nhọn chiếm vỉa hè gây tai nạn nguy kịch",
             "Hộ gia đình tự ý hàn khung sắt có chông nhọn nhô ra vỉa hè chiếm lối đi, một cháu nhỏ chạy bộ va vào rách da đứt mạch máu cấp cứu, đề nghị cưỡng chế tháo dỡ khẩn cấp."),
            ("Bãi giữ xe tự phát chiếm toàn bộ vỉa hè ép người đi bộ xuống lòng đường bị xe đâm",
             "Bãi giữ xe trái phép chặn hết lối đi bộ, một cụ già buộc phải đi bộ dưới lòng đường thì bị xe máy đâm ngã gãy chân, đề nghị đình chỉ và xử lý nghiêm."),
        ],
        "normal": [
            ("Hàng quán cà phê kê bàn ghế chiếm vỉa hè đẩy người đi bộ xuống lòng đường",
             "Quán cà phê kê bàn ghế tràn lan kín toàn bộ vỉa hè từ sáng đến tối, người đi bộ không còn lối đi phải bước xuống lòng đường nguy hiểm, đề nghị phường ra quân xử lý."),
            ("Ô tô đỗ chắn ngang vỉa hè và cửa ra vào của ngõ dân cư",
             "Nhiều xe ô tô vô ý thức đậu ngang trên vỉa hè bít kín lối đi của ngõ xóm khiến người đi xe máy dắt xe ra vào rất khó khăn, đề nghị dán thông báo xử phạt."),
            ("Biển quảng cáo di động đặt tràn lan dưới lòng lề đường cản trở giao thông",
             "Các cửa hàng kinh doanh đặt biển hiệu hộp đèn chằng chịt dưới mép đường che khuất tầm nhìn của các phương tiện rẽ nhánh, đề nghị thu giữ biển hiệu vi phạm."),
            ("Xây bậc tam cấp kiên cố lấn ra vỉa hè công cộng gây vấp ngã",
             "Nhiều nhà mặt tiền tự ý xây bậc dắt xe bằng bê tông lồi ra mặt vỉa hè hơn nửa mét làm hẹp lối đi chung, đề nghị yêu cầu đập bỏ hoàn trả mặt bằng."),
            ("Tập kết vật liệu cát đá xây dựng chiếm dụng vỉa hè suốt nhiều ngày",
             "Công trình tư nhân đổ cát đá xi măng chiếm trọn vỉa hè không chừa một lối đi nào, cát trôi xuống cống khi mưa, đề nghị xử phạt vi phạm trật tự đô thị."),
        ],
        "low": [
            ("Kiến nghị kẻ vạch sơn phân định ranh giới vỉa hè rõ ràng cho kinh doanh",
             "Đề xuất UBND phường triển khai kẻ vạch sơn vàng chừa lại tối thiểu 1.5m vỉa hè thông thoáng cho người đi bộ, phần bên trong cho phép để xe máy có trật tự."),
            ("Đề xuất lắp đặt các thanh chắn barie ngăn xe máy phóng lên vỉa hè",
             "Kiến nghị bố trí cọc tiêu ziczac ở đầu tuyến vỉa hè để ngăn chặn xe máy leo lề vào giờ cao điểm, bảo đảm an toàn cho người tập thể dục."),
            ("Góp ý xây dựng tuyến phố kiểu mẫu văn minh đô thị không lấn chiếm",
             "Đề xuất phát động cuộc thi tuyến phố xanh - sạch - đẹp không vi phạm hành lang an toàn giao thông giữa các phường trong quận."),
            ("Kiến nghị quy hoạch chợ đêm ẩm thực tập trung để giải tỏa hàng rong lề đường",
             "Đề xuất địa phương quy hoạch một khu đất trống làm điểm buôn bán tập trung cho các hộ kinh doanh thức ăn đường phố."),
        ],
    },
    "public_procedures": {
        "unit": "intake",
        "high": [
            ("Cán bộ tiếp dân vòi vĩnh đòi hối lộ tiền phong bì mới chịu giải quyết hồ sơ",
             "Người dân đi làm thủ tục cấp sổ đỏ bị cán bộ yêu cầu đưa 50 triệu tiền ngoài luồng mới cho ký giấy hẹn, có ghi âm bằng chứng vi phạm pháp luật nghiêm trọng, đề nghị thanh tra và công an vào cuộc khẩn cấp."),
            ("Cán bộ cố tình làm thất lạc hồ sơ gốc của công dân gây hậu quả nghiêm trọng",
             "Bộ hồ sơ gốc thừa kế tài sản công dân nộp tại bộ phận một cửa bị cán bộ làm mất nhưng chối bỏ trách nhiệm và đùn đẩy, làm lỡ thời hạn khởi kiện của gia đình, đề nghị kiểm tra xử lý khẩn cấp."),
            ("Hệ thống cổng dịch vụ công quốc gia bị sập đúng hạn nộp hồ sơ thuế quan trọng",
             "Cổng dịch vụ công tê liệt hoàn toàn báo lỗi hệ thống khiến hàng trăm doanh nghiệp không thể nộp tờ khai đúng thời hạn cuối cùng của tháng có nguy cơ bị phạt oan, đề nghị kỹ thuật khắc phục khẩn cấp."),
        ],
        "normal": [
            ("Hồ sơ đất đai trễ hẹn trả kết quả gần 2 tháng không rõ lý do",
             "Tôi nộp hồ sơ xin cấp đổi giấy chứng nhận quyền sử dụng đất từ tháng trước, giấy hẹn ghi ngày trả kết quả nhưng đến nay đã quá hạn gần 2 tháng mà không có văn bản giải thích, đề nghị kiểm tra đôn đốc."),
            ("Cán bộ bộ phận một cửa có thái độ gắt gỏng cửa quyền với người già",
             "Khi người lớn tuổi đến làm thủ tục xác nhận tình trạng hôn nhân hỏi hướng dẫn khai mẫu đơn thì cán bộ tiếp nhận to tiếng nạt nộ thiếu văn hóa ứng xử, đề nghị chấn chỉnh tác phong phục vụ."),
            ("Thủ tục hành chính yêu cầu thêm nhiều loại giấy tờ không đúng quy định",
             "Theo quy định hiện hành đã bỏ sổ hộ khẩu giấy nhưng cán bộ vẫn đòi hỏi công dân phải xuất trình xác nhận cư trú giấy mới tiếp nhận hồ sơ, đề nghị chỉ đạo thống nhất quy trình."),
            ("Ứng dụng dịch vụ công trực tuyến báo lỗi không tải được file đính kèm",
             "Tôi làm hồ sơ đăng ký kinh doanh qua mạng nhưng đến bước tải tệp PDF CCCD thì hệ thống liên tục báo lỗi kết nối máy chủ, đề nghị kiểm tra hạ tầng phần mềm."),
            ("Thời gian chờ đợi giải quyết thủ tục tại bộ phận một cửa quá lâu",
             "Người dân lấy số thứ tự từ sáng sớm nhưng đến trưa vẫn chưa tới lượt do chỉ mở 1 quầy giao dịch cho thủ tục cấp phép xây dựng, đề nghị bố trí thêm cán bộ hỗ trợ."),
        ],
        "low": [
            ("Kiến nghị đơn giản hóa các bước kê khai thủ tục trực tuyến trên Cổng dịch vụ công",
             "Đề xuất Sở Thông tin và Truyền thông tối ưu hóa giao diện nộp hồ sơ trực tuyến, giảm bớt các trường thông tin trùng lặp để người dân dễ sử dụng hơn."),
            ("Đề xuất mở lớp tập huấn hướng dẫn người cao tuổi sử dụng ứng dụng công dân số",
             "Kiến nghị đoàn thanh niên phối hợp hội người cao tuổi phường mở các buổi hướng dẫn cài đặt và sử dụng ứng dụng hành chính công trên điện thoại thông minh."),
            ("Góp ý bổ sung thêm hình thức thanh toán lệ phí trực tuyến qua ví điện tử",
             "Đề xuất cổng thanh toán tích hợp đa dạng các cổng như Viettel Money, VNPT Money, MoMo để người dân thuận tiện nộp phí."),
            ("Kiến nghị kéo dài giờ tiếp nhận hồ sơ vào sáng thứ 7 cho người đi làm",
             "Đề xuất bộ phận tiếp nhận và trả kết quả mở thêm ca làm việc sáng thứ bảy hàng tuần để tạo điều kiện thuận lợi cho công nhân viên chức."),
        ],
    },
    "security_order_disputes": {
        "unit": "intake",
        "high": [
            ("Hai nhóm thanh niên mang hung khí dao kiếm tụ tập hỗn chiến",
             "Khoảng 20 thanh niên đi xe máy cầm theo tuýp sắt gắn dao phóng lợn và đao kiếm tụ tập la hét chửi thề chuẩn bị xông vào đâm chém nhau, đề nghị lực lượng cảnh sát 113 đến hiện trường khẩn cấp."),
            ("Bắt quả tang đối tượng cạy cửa đột nhập trộm cắp tài sản đang chống trả",
             "Kẻ gian cạy khóa vào nhà dân trộm đồ bị phát hiện thì rút dao bấm chống trả đe dọa đâm chủ nhà, bà con đang vây bắt xung quanh, đề nghị công an phường đến hỗ trợ khẩn cấp."),
            ("Đòi nợ thuê ném chất bẩn mắm tôm và sơn đỏ đe dọa đốt nhà dân",
             "Nhóm đối tượng lạ mặt ném sơn đỏ và dầu luyn pha mắm tôm vào cửa nhà dân, để lại thư đe dọa bắt cóc con cái và đốt nhà nếu không trả nợ thay người thân, đề nghị bảo vệ khẩn cấp."),
            ("Chồng bạo hành đánh đập vợ dã man đe dọa dùng vũ lực gây nguy hiểm",
             "Người đàn ông say rượu đang cầm hung khí đánh đập người vợ và đập phá đồ đạc trong nhà, tiếng kêu cứu thảm thiết vang cả ngõ xóm, đề nghị công an can thiệp khẩn cấp bảo vệ nạn nhân."),
        ],
        "normal": [
            ("Hàng xóm to tiếng cãi vã tranh chấp ranh giới xây dựng gây mất trật tự",
             "Hai hộ gia đình sát vách mâu thuẫn gay gắt về mốc ranh giới giọt ranh mái tôn, thường xuyên to tiếng cãi vã chửi bới xúc phạm nhau làm ầm ĩ khu dân cư, đề nghị tổ hòa giải xuống can thiệp."),
            ("Tình trạng trộm cắp vặt cây cảnh và gương xe máy thường xuyên xảy ra",
             "Dạo gần đây trong ngõ xuất hiện kẻ gian lợi dụng lúc vắng người bẻ gương xe máy và nhổ trộm cây cảnh trước cửa nhà dân, đề nghị công an tăng cường tuần tra."),
            ("Tranh chấp chỗ đỗ xe ô tô trong ngõ ngách dẫn đến chửi bới xúc phạm",
             "Một số người ở nơi khác đến đậu xe bịt lối đi của các hộ trong ngõ, khi nhắc nhở thì xảy ra cãi cọ va chạm lời qua tiếng lại gây căng thẳng, đề nghị khu phố có biển hướng dẫn."),
            ("Nhóm thiếu niên tụ tập hút thuốc điện tử gây mất trật tự công viên",
             "Vào buổi tối tại hoa viên công cộng có nhóm thanh thiếu niên tụ tập nói tục chửi bậy và xả rác bừa bãi, đề nghị lực lượng dân phòng tuần tra nhắc nhở."),
            ("Nuôi chó thả rông không rọ mõm phóng uế và đuổi cắn người đi đường",
             "Hộ dân nuôi nhiều chó dữ thả rông ngoài đường không đeo rọ mõm khiến trẻ em đi học sợ hãi, chó đã cắn rách quần người đi bộ, đề nghị xử phạt chủ nuôi."),
        ],
        "low": [
            ("Kiến nghị công an phường tăng cường tuần tra đêm ngõ xóm phòng ngừa tội phạm",
             "Đề xuất lực lượng công an phường và dân quân tự vệ phối hợp lập chốt tuần tra lưu động vào khung giờ từ 0h đến 3h sáng tại các khu vực ngõ cụt vắng vẻ."),
            ("Đề xuất lắp đặt camera an ninh giám sát toàn diện tại các nút giao ngõ xóm",
             "Kiến nghị hội đồng nhân dân phường thông qua đề án xã hội hóa lắp đặt camera an ninh kết nối về trụ sở công an phường để phòng chống tội phạm."),
            ("Góp ý nâng cao chất lượng hoạt động của các tổ hòa giải cơ sở",
             "Đề xuất tổ chức tập huấn kiến thức pháp luật và kỹ năng hòa giải cho các bác tổ trưởng dân phố để xử lý êm thấm các xích mích nhỏ trong xóm làng."),
            ("Đề xuất tổ chức phong trào toàn dân bảo vệ an ninh Tổ quốc tại khu dân cư",
             "Kiến nghị địa phương phát động phong trào tự quản, trao thưởng kịp thời cho những công dân dũng cảm phát hiện và tố giác tội phạm."),
        ],
    },
}

PREFIX_TEMPLATES = [
    "{title}",
    "[{urgency_tag}] {title}",
    "Phản ánh: {title} tại {district}",
    "{province} - {title}",
    "Khẩn thiết xử lý: {title}",
    "Đề nghị cơ quan chức năng: {title}",
    "Ý kiến cử tri: {title}",
    "Thông tin phản ánh hiện trường: {title}",
]

EMOTION_STARTERS = [
    "Tôi xin phản ánh sự việc cụ thể như sau: ",
    "Kính gửi các cấp chính quyền và cơ quan liên quan: ",
    "Người dân chúng tôi vô cùng bức xúc và kính mong được giải quyết: ",
    "Phản ánh đến tổng đài tiếp nhận thông tin đô thị: ",
    "Ghi nhận tình trạng thực tế tại địa bàn: ",
    "Bà con nhân dân tại khu vực đề nghị cơ quan chức năng khẩn trương: ",
    "",
]


def generate_dataset(total_samples=36000):
    samples_per_cat = total_samples // 12
    samples_per_urg = samples_per_cat // 3
    print(f"Target: {total_samples} total samples ({samples_per_cat}/category, {samples_per_urg}/urgency per category)")

    samples = []
    sample_id = 1
    now = datetime(2026, 11, 13, 23, 0, 0)

    for cat_name, cat_data in CATEGORIES_PATTERNS.items():
        unit = cat_data["unit"]
        for urg_level in ["high", "normal", "low"]:
            pattern_list = cat_data[urg_level]
            urgency_tag = urg_level.upper()

            for _ in range(samples_per_urg):
                base_title, base_desc = random.choice(pattern_list)
                province = random.choice(PROVINCES_LIST)
                lat_min, lat_max, lon_min, lon_max = PROVINCES_GEO[province]
                lat = round(random.uniform(lat_min, lat_max), 5)
                lon = round(random.uniform(lon_min, lon_max), 5)

                district = random.choice(DISTRICTS)
                street = random.choice(STREETS)
                street_num = random.randint(1, 999)

                title_template = random.choice(PREFIX_TEMPLATES)
                title = title_template.format(
                    title=base_title,
                    urgency_tag=urgency_tag,
                    district=district,
                    province=province,
                )

                starter = random.choice(EMOTION_STARTERS)
                address_str = f"số {street_num} {street}, {district}, {province}"
                full_description = f"{starter}{base_desc} (Địa chỉ ghi nhận: {address_str})."
                full_text = f"{province} - {district} - {title}. {full_description}"

                # Random timestamp in the last 30 days
                delta_days = random.randint(0, 30)
                delta_hours = random.randint(0, 23)
                delta_minutes = random.randint(0, 59)
                timestamp_dt = now - timedelta(days=delta_days, hours=delta_hours, minutes=delta_minutes)
                iso_time = timestamp_dt.isoformat() + "Z"

                samples.append({
                    "id": f"vn63-mt-{sample_id:06d}",
                    "city": province,
                    "district": district,
                    "street": street,
                    "latitude": lat,
                    "longitude": lon,
                    "created_at": iso_time,
                    "label_12": cat_name,
                    "label_4": unit,
                    "urgency": urg_level,
                    "title": title,
                    "description": full_description,
                    "full_text": full_text,
                })
                sample_id += 1

    random.shuffle(samples)

    json_path = Path("d:/mnm/civic_dataset_multitask_nationwide.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    csv_path = Path("d:/mnm/civic_dataset_multitask_nationwide.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "city",
                "district",
                "street",
                "latitude",
                "longitude",
                "created_at",
                "label_12",
                "label_4",
                "urgency",
                "title",
                "description",
                "full_text",
            ],
        )
        writer.writeheader()
        writer.writerows(samples)

    print(f"Generated {len(samples)} samples successfully!")
    print(f"Files saved: {json_path} and {csv_path}")


if __name__ == "__main__":
    generate_dataset(36000)

